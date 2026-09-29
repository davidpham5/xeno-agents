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
} from "./agent-data";
import { writeReport, chooseAction } from "./agent-brain";
import { checkoutServiceUrl } from "./observe";

import { logAgentActivity } from "./agent-log";
import { actionPolicy } from "./tool-policy";
import type { ActionName } from "../shared/types";
import { environments } from "./schema";

async function executeAction(
  environmentId: string,
  runId: string,
  actionId: string,
  name: ActionName,
  expectedVersion?: number,
) {
  const input = expectedVersion === undefined ? {} : { expectedVersion };

  // log an performed activity
  await logAgentActivity(environmentId, runId, "act", `Calling ${name}`, {
    actionId,
    input,
  });

  try {
    const response = await fetch(`${checkoutServiceUrl}/operations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actionId,
        name,
        ...input,
      }),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(String(result.error || "Tool call failed"));

    if (result.stale !== true) {
      await recordToolAction(environmentId, runId, actionId, name, input);
    }
  } catch (e) {
    await logAgentActivity(
      environmentId,
      runId,
      "act",
      `${name} attempt failed`,
      {
        actionId,
        error: e.message ?? String(e),
      },
    );
  }
}

// The alert intake has already opened an incident. Build its harness here.
export const incidentAgent = inngest.createFunction(
  {
    id: "incident-agent",
    name: "Checkout incident agent",
    triggers: { event: "incident/opened" },
  },

  async ({ event, step }) => {
    const { environmentId, runId, instanceId } = event.data;
    const run = await getRun(runId);

    if (
      run.instanceId !== instanceId ||
      ["completed", "failed", "cancelled", "escalated", "superseded"].includes(
        run.status,
      )
    ) {
      return;
    }

    // at the top of every single loop, what we want to do and add a checkpoint?
    for (let cycle = 1; cycle <= 24; cycle++) {
      // get current run
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
      const state = await step.run(`observer-state-${cycle}`, () => {
        return agentState(environmentId, runId);
      });

      if (goalSatisfied(state, run.goalCondition)) {
        const report = await step.run(`write-report-${cycle}`, () =>
          writeReport(run.goal, state),
        );
        await step.run(`complete-run${cycle}`, () =>
          setRun(runId, "completed", null, report),
        );
        // await setRun(runId, "completed", null, report);
        return { report };
      }
    }

    return step.run("whole-agentLoop", async () => {
      const run = await getRun(runId);
      // this agent is running in the background and observing the state
      // harness determining checking if it satisfy the goal
      for (let cycle = 1; cycle <= 8; cycle++) {
        // this pattern resembes a state reducer like in redux.
        const state = await agentState(environmentId, runId);
        if (goalSatisfied(state, run.goalCondition)) {
          const report = await writeReport(run.goal, state);
          await setRun(runId, "completed", null, report);
          return { report };
        }

        const decision = await chooseAction(run.goal, state);
        await setIteration(runId, cycle);
        // how do i describe the state of the world vividly?
        // logs it to the harness
        await recordDecision(
          environmentId,
          runId,
          cycle,
          decision.action,
          decision.reason,
        );
        // let's handle exceptions where it needs our help.
        if (
          decision.action === "wait" ||
          decision.action === "complete" ||
          decision.action === "request_help"
        ) {
          await setRun(
            runId,
            "escalated",
            "this first loop cannot pause just yet",
          );
          return;
        }

        if (decision.action === "rollback_release") {
          // for now don't run the rollback
          await setRun(runId, "escalated", "Approvals not build yet");
        }

        // any decision than these four defined decisions, let it rip

        const actionId = `${runId}: ${randomUUID}`; // we want to be super clinical with trusting our agent. we need recipts
        // keep the business logic on the service itself. just perform the action for us --> /operations
        const response = await fetch(checkoutServiceUrl + "/operations", {
          // this is how we actually change the state is changed
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            actionId,
            name: decision.action,
            expectedVersion: state.world.version, // version our state like we version our api
          }),
        });

        // need to confirm to the Harness something happened
        const result = await response.json();
        if (!response.ok)
          throw new Error(String(result.error ?? response.status));
        if (result.stale !== true) {
          await recordToolAction(
            environmentId,
            runId,
            actionId,
            decision.action,
            {
              expectedVersion: state.world.version,
            },
            result,
          );
        }
      }
      // update the run
      await setRun(runId, "escalated", "Decision limit reached");
    });
  },
);
