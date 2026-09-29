// =============================================================================
// AGENT WORKFLOW — the "harness" that runs an incident-response agent
// =============================================================================
//
// Big picture: when an incident opens, we run an agent in a loop. Each pass of
// the loop follows the classic agent cycle:
//
//     OBSERVE  ->  DECIDE  ->  ACT  ->  (wait a beat)  ->  repeat
//        |           |          |
//        |           |          └─ call a real tool on the checkout service
//        |           └─ ask the LLM which single action to take next
//        └─ read the current world state (service status, recent events, etc.)
//
// The agent keeps looping until it reaches the goal, decides it's done, needs a
// human, or hits a safety limit on the number of decisions.
//
// Two roles to keep straight while reading:
//   • The LLM ("the brain", in agent-brain.ts) only *suggests* one action.
//   • This file ("the harness") *decides whether and how* to run it. The harness
//     owns all authority: policy checks, approvals, retries, and recording what
//     happened. Never let the model act directly — it proposes, the harness disposes.
//
// This runs on Inngest, which gives us "durable execution": the function can
// pause, sleep, crash, and resume without re-doing work. The unit of durability
// is `step.run(...)` — see the note further down about why almost everything is
// wrapped in a step.
// =============================================================================

import { randomUUID } from "node:crypto";
import { inngest } from "./inngest";
import {
  agentState,
  getRun,
  goalSatisfied,
  recordDecision,
  recordToolAction,
  setIteration,
  setRun,
  staleProposal,
  getProposal,
  proposeAction,
} from "./agent-data";
import { writeReport, chooseAction } from "./agent-brain";
import { checkoutServiceUrl } from "./observe";

import { logAgentActivity } from "./agent-log";
import { actionPolicy } from "./tool-policy";
import type { ActionName } from "../shared/types";

// -----------------------------------------------------------------------------
// executeAction — the agent's "hands".
//
// This is the only place we actually reach out and change (or read) the real
// checkout service. It POSTs the chosen action to the service's /operations
// endpoint and records what came back. The business logic lives on the service;
// here we just call it, then log the result.
//
// `expectedVersion` is an optimistic-concurrency check. For write actions we
// send the version of the world we based our decision on. If the service moved
// on since we observed it, it replies `stale: true` and refuses — so we never
// act on a stale picture of reality. Read actions pass no version.
// -----------------------------------------------------------------------------
async function executeAction(
  environmentId: string,
  runId: string,
  actionId: string,
  name: ActionName,
  expectedVersion?: number,
) {
  // For reads there is no version to guard against; for writes we attach the
  // version we expect the world to still be at.
  const input = expectedVersion === undefined ? {} : { expectedVersion };

  // Record that we are *about* to call the tool (before we know the outcome).
  await logAgentActivity(
    environmentId,
    runId,
    "act",
    `Calling ${name.replaceAll("_", " ")}`,
    {
      actionId,
      input,
    },
  );

  try {
    // Perform the action on the real service. `actionId` is sent along so the
    // service can recognize duplicate calls (idempotency) if we ever retry.
    const response = await fetch(`${checkoutServiceUrl}/operations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actionId,
        name,
        ...input,
      }),
    });
    const result = (await response.json()) as Record<string, unknown>;
    if (!response.ok)
      throw new Error(
        String(result.error || `Tool failed: ${response.status}`),
      );

    // Only record a "real" action. If the service said `stale`, nothing changed,
    // so there is no effect worth writing to our action history.
    if (result.stale !== true) {
      await recordToolAction(
        environmentId,
        runId,
        actionId,
        name,
        input,
        result,
      );
    }
    return result;
  } catch (e) {
    // If the call blew up, note the failure and re-throw so the surrounding
    // step.run can retry (Inngest retries failed steps for us).
    await logAgentActivity(
      environmentId,
      runId,
      "act",
      `${name.replaceAll("_", " ")} attempt failed`,
      {
        actionId,
        error: e instanceof Error ? e.message : String(e),
      },
    );
    throw e;
  }
}

// -----------------------------------------------------------------------------
// incidentAgent — the durable Inngest function that IS the agent.
//
// The alert intake has already opened an incident and created a "run" row in the
// DB. This function is the harness that drives that run to completion.
//
// The config object below tells Inngest how to manage this function:
//   • triggers    — run whenever an "incident/opened" event fires.
//   • retries: 2  — if the function throws, Inngest re-runs it up to 2 more times.
//                   (Completed steps are cached, so retries resume, not restart.)
//   • concurrency — at most 1 of these runs *executing at a time per environment*.
//                   Incidents in the same environment queue up instead of racing
//                   each other. This caps parallel execution, not how many can wait.
//   • onFailure   — the safety net: if all retries are exhausted, mark the run
//                   "failed" so it doesn't sit stuck in "running" forever.
// -----------------------------------------------------------------------------
export const incidentAgent = inngest.createFunction(
  {
    id: "incident-agent",
    name: "Checkout incident agent",
    triggers: { event: "incident/opened" },
    retries: 2,
    // This limits executing steps, not the number of waiting incidents.
    concurrency: { limit: 1, key: "event.data.environmentId" },
    onFailure: async ({ error, event }) => {
      // On terminal failure, pull the runId out of the original event and record
      // the failure on the run so the UI (and humans) can see it.
      const original = event.data.event as { data?: { runId?: string } };
      if (original.data?.runId)
        await setRun(original.data.runId, "failed", error.message);
    },
  },

  // `event` carries the incident data; `step` is Inngest's toolkit for durable
  // steps (step.run, step.sleep, ...). Everything with a side effect goes through
  // a step so it survives retries and resumes.
  async ({ event, step }) => {
    const { environmentId, runId, instanceId } = event.data;
    const run = await getRun(runId);

    // Guard before we start: bail out if this run is stale.
    //   • instanceId mismatch = the checkout service restarted since the incident
    //     opened, so this run is talking about a world that no longer exists.
    //   • an already-terminal status = someone/something finished it already.
    if (
      run.instanceId !== instanceId ||
      ["completed", "failed", "cancelled", "escalated", "superseded"].includes(
        run.status,
      )
    ) {
      return;
    }

    // THE AGENT LOOP. Each `cycle` is one OBSERVE -> DECIDE -> ACT pass.
    // Capped at 24 decisions so a confused agent can't loop forever (see the
    // "Decision limit reached" escalation after the loop).
    for (let cycle = 1; cycle <= 24; cycle++) {
      // Re-read the run each pass: a human may have cancelled it, or another
      // process may have moved it to a terminal state between cycles.
      const current = await getRun(runId);
      if (
        [
          "completed",
          "failed",
          "cancelled",
          "escalated",
          "superseded",
        ].includes(current.status)
      ) {
        return;
      }

      // === OBSERVE ===
      // Snapshot the world: current service state + recent events/actions. This
      // is the deterministic "current state" the agent reasons over — like the
      // state passed into a reducer. The agent doesn't get to choose what's true;
      // it only gets to read it.
      const state = await step.run(`observe-state-${cycle}`, () => {
        return agentState(environmentId, runId);
      });

      // Cheap deterministic exit BEFORE spending an LLM call: if the configured
      // goal condition is already met (e.g. service state === "healthy"), write
      // the final report and finish. No need to ask the model anything.
      if (goalSatisfied(state, run.goalCondition)) {
        const report = await step.run(`write-report-${cycle}`, () =>
          writeReport(run.goal, state),
        );
        await step.run(`complete-run-${cycle}`, () =>
          setRun(runId, "completed", null, report),
        );
        return { report };
      }

      // === DECIDE ===
      // Ask the LLM for exactly ONE next action + reason (see chooseAction /
      // decisionSchema in agent-brain.ts — structured output, so we get a typed
      // object back, not free text).
      //
      // Why wrap it in step.run? Two reasons:
      //   1) It costs money/tokens, so we don't want to re-run it on every retry.
      //   2) The LLM is non-deterministic — same input can give a different answer.
      // A step caches its result, so once this cycle's decision is made it stays
      // fixed even if a later step fails and Inngest replays the function. Mental
      // model: a step is like React's useMemo — compute the expensive/unstable
      // thing once, then reuse the memoized value.
      const decision = await step.run(`choose-action-${cycle}`, () =>
        chooseAction(run.goal, state),
      );

      // Persist which cycle we're on, then log the decision for the timeline UI.
      await step.run(`set-iteration-${cycle}`, () =>
        setIteration(runId, cycle),
      );
      await step.run(`record-decision-${cycle}`, () =>
        recordDecision(
          environmentId,
          runId,
          cycle,
          decision.action,
          decision.reason,
        ),
      );

      // === ACT (with harness authority) ===
      // From here down the harness decides whether the model's chosen action is
      // actually allowed to run. Several actions aren't fully built yet, so we
      // escalate (hand off to a human) instead of pretending we handled them.

      // "wait", "request_help", and goal-gated "complete" all mean the same thing:
      // the agent has no safe action to take right now. Instead of escalating, we
      // SUSPEND the run and wait for the world to change (a new service event),
      // then re-observe and decide again. This is the heart of lesson 4.
      if (
        decision.action === "wait" ||
        decision.action === "request_help" ||
        (decision.action === "complete" && run.goalCondition)
      ) {
        // The model said "complete" but a goal condition is configured and not
        // yet met — log that we're overriding it so it can't falsely finish.
        if (decision.action === "complete") {
          await step.run(`reject-completion-${cycle}`, () =>
            logAgentActivity(
              environmentId,
              runId,
              "human",
              "Completion blocked by configured goal condition",
              { condition: run.goalCondition },
            ),
          );
        }
        // Human help isn't built yet, so we note the question and fall through to
        // the same wait-for-service-evidence loop below.
        if (decision.action === "request_help") {
          await step.run(`defer-help-${cycle}`, () =>
            logAgentActivity(
              environmentId,
              runId,
              "human",
              "Help is not available yet",
              { question: decision.detail },
            ),
          );
        }
        // Mark the run "waiting" (this is the state you'll see as sleeping in the
        // Inngest dashboard), with a reason that fits which action we're waiting on.
        await step.run(`wait-status-${cycle}`, () =>
          setRun(
            runId,
            "waiting",
            decision.action === "request_help"
              ? "Help is not available yet; waiting for fresh service evidence"
              : decision.action === "wait"
                ? decision.reason
                : "Recovery is not verified",
          ),
        );

        // THE WAIT LOOP. Park here using NO compute until the world changes.
        let check = 0;
        while (true) {
          // Re-read the run. If a human cancelled/superseded/failed it while we
          // slept, stop the whole function.
          const latest = await step.run(
            `read-event-sequence-${cycle}-${check}`,
            () => getRun(runId),
          );
          if (["cancelled", "superseded", "failed"].includes(latest.status))
            return;
          // The world moved since we last observed (fresh evidence arrived) →
          // break out so the for-loop re-observes with the new state.
          if (latest.eventSequence > state.eventSequence) break;
          check++;
          // Sleep (zero compute) until an "incident/updated" event for THIS run
          // arrives, or 10s elapses. On timeout we just loop and re-check.
          await step.waitForEvent(`wait-for-service-${cycle}-${check}`, {
            event: "incident/updated",
            if: `async.data.runId == "${runId}"`,
            timeout: "10s",
          });
          // The timeout also reconciles an event that arrived just before the wait.
        }
        // Back to the top of the for-loop → OBSERVE fresh state → DECIDE again.
        continue;
      }

      // "complete" with NO goal condition configured: we trust the model's
      // judgment that it's done, write the report, and finish.
      if (decision.action === "complete") {
        const report = await step.run(`write-report-${cycle}`, () =>
          writeReport(run.goal, state),
        );

        await step.run(`complete-run-${cycle}`, () =>
          setRun(runId, "completed", null, report),
        );

        return { report };
      }

      // (Note: "request_help" no longer escalates here — it's handled in the
      // wait branch above, which suspends until fresh service evidence arrives.)

      // Unique ID for this specific attempt, used as an idempotency key by the
      // service. (Scoped per attempt, so if a response is lost we may retry as a
      // fresh action rather than dedupe — a known tradeoff, not an oversight.)
      const actionId = `${runId}:${randomUUID()}`;

      // Look up the harness policy for the chosen action (see tool-policy.ts).
      // This is the "who has authority?" gate: read = safe to run, write = run it,
      // approval = a human must sign off first. The model proposes; policy disposes.
      const policy = actionPolicy[decision.action];

      // Destructive/high-risk actions (e.g. rollback_release) require approval.
      // The approval gate isn't built, so escalate instead of acting unilaterally.
      if (policy === "approval") {
        // await step.run(`approval-unavailable-${cycle}`, () =>
        //   setRun(runId, "escalated", "Approval gate is not built yet"),
        // );
        const input = { expectedVersion: state.world.version }; //snapshot of the state
        const proposalId = await step.run(
          `propose-action-${cycle}`,
          async () => {
            const proposal = await proposeAction(
              environmentId,
              runId,
              actionId,
              decision.action,
              input,
            );
            return proposal.id;
          },
        );
        // why create a proposal to only get the id and then here
        // and then make a proposal here?
        // because proposal overwrite. Get variable from DB and not cache value at some point
        // Getting stale proposals from the above will lead to a bug
        let proposal = await step.run(`read-human-${cycle}`, () =>
          getProposal(proposalId),
        );

        let check = 0;

        while (proposal.status === "pending") {
          check++;
          await step.waitForEvent(`wait-for-human-${cycle}`, {
            event: "agent/approval.decided",
            if: `async.data.proposalId == "${proposalId}"`,
            timeout: "10s",
          });

          proposal = await step.run(`reconcile-human-${cycle}-${check}`, () =>
            getProposal(proposalId),
          );
        }
        if (proposal.status !== "approved") {
          await step.run(`stop-after-human-${cycle}`, () =>
            setRun(runId, "escalated", `Human decision: ${proposal.status}`),
          );
          return;
        }

        const fresh = await step.run(`recheck-approved-state-${cycle}`, () =>
          agentState(environmentId, runId),
        );

        if (fresh.world.version !== input.expectedVersion) {
          await step.run(`invalidate-approval-${cycle}`, () =>
            staleProposal(proposalId),
          );
          continue;
        }
      }

      // Actually run the tool. For writes we pass the observed world version so
      // the service can reject the call if the world changed underneath us
      // (optimistic concurrency). Reads pass no version.
      const result = await step.run(`execute-action-${cycle}`, () => {
        return executeAction(
          environmentId,
          runId,
          actionId,
          decision.action,
          policy === "read" ? undefined : state.world.version,
        );
      });

      // The service refused because its state moved since we observed it. Nothing
      // changed, so just log it and loop again — the next cycle re-observes the
      // fresh state and decides anew. `continue` skips the settle/sleep below.
      if (result.stale === true) {
        await step.run(`stale-action-${cycle}`, () =>
          logAgentActivity(
            environmentId,
            runId,
            "act",
            "Action rejected because service state changed",
            result,
          ),
        );
        continue;
      }

      // The action ran and changed the world. Give the service a moment to
      // "settle" before we observe again next cycle, so we don't read the effect
      // half-applied. step.sleep is durable: Inngest actually parks the function
      // and uses NO compute while sleeping, then wakes it to run the next cycle.
      await step.run(`settle-status-${cycle}`, () =>
        setRun(runId, "waiting", "Waiting briefly before observing the effect"),
      );
      await step.sleep(
        `settle-${cycle}`,
        process.env.AGENT_SETTLE_DELAY ?? "1s",
      );
    }

    // Fell out of the loop = we hit the 24-decision cap without reaching the goal.
    // Escalate to a human rather than silently spinning forever.
    await step.run("stop-at-limit", () =>
      setRun(runId, "escalated", "Decision limit reached"),
    );

    // Previous single-step version of the loop, kept for reference (replaced by the stepped loop above).
    // return step.run("whole-agentLoop", async () => {
    //   const run = await getRun(runId);
    //   // this agent is running in the background and observing the state
    //   // harness determining checking if it satisfy the goal
    //   for (let cycle = 1; cycle <= 8; cycle++) {
    //     // this pattern resembes a state reducer like in redux.
    //     const state = await agentState(environmentId, runId);
    //     if (goalSatisfied(state, run.goalCondition)) {
    //       const report = await writeReport(run.goal, state);
    //       await setRun(runId, "completed", null, report);
    //       return { report };
    //     }
    //
    //     const decision = await chooseAction(run.goal, state);
    //     await setIteration(runId, cycle);
    //     // how do i describe the state of the world vividly?
    //     // logs it to the harness
    //     await recordDecision(
    //       environmentId,
    //       runId,
    //       cycle,
    //       decision.action,
    //       decision.reason,
    //     );
    //     // let's handle exceptions where it needs our help.
    //     if (
    //       decision.action === "wait" ||
    //       decision.action === "complete" ||
    //       decision.action === "request_help"
    //     ) {
    //       await setRun(
    //         runId,
    //         "escalated",
    //         "this first loop cannot pause just yet",
    //       );
    //       return;
    //     }
    //
    //     if (decision.action === "rollback_release") {
    //       // for now don't run the rollback
    //       await setRun(runId, "escalated", "Approvals not build yet");
    //     }
    //
    //     // any decision than these four defined decisions, let it rip
    //
    //     const actionId = `${runId}: ${randomUUID}`; // we want to be super clinical with trusting our agent. we need recipts
    //     // keep the business logic on the service itself. just perform the action for us --> /operations
    //     const response = await fetch(checkoutServiceUrl + "/operations", {
    //       // this is how we actually change the state is changed
    //       method: "POST",
    //       headers: { "Content-Type": "application/json" },
    //       body: JSON.stringify({
    //         actionId,
    //         name: decision.action,
    //         expectedVersion: state.world.version, // version our state like we version our api
    //       }),
    //     });
    //
    //     // need to confirm to the Harness something happened
    //     const result = await response.json();
    //     if (!response.ok)
    //       throw new Error(String(result.error ?? response.status));
    //     if (result.stale !== true) {
    //       await recordToolAction(
    //         environmentId,
    //         runId,
    //         actionId,
    //         decision.action,
    //         {
    //           expectedVersion: state.world.version,
    //         },
    //         result,
    //       );
    //     }
    //   }
    //   // update the run
    //   await setRun(runId, "escalated", "Decision limit reached");
    // });
  },
);
