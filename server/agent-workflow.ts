import { randomUUID } from "node:crypto";
import { inngest } from "./inngest";
import {
  agentState,
  getRun,
  goalSatisfied,
  recordToolAction,
  setIteration,
  setRun,
} from "./agent-data";
import { writeReport } from "./agent-brain";

// The alert intake has already opened an incident. Build its harness here.
export const incidentAgent = inngest.createFunction(
  {
    id: "incident-agent",
    name: "Checkout incident agent",
    triggers: { event: "incident/opened" },
  },
  // async ({ event, step }) => {
  //   await setRun(event.data.runId, 'escalated', 'Harness not implemented yet')
  //   return { runId: event.data.runId, next: 'Build the observe–decide–act loop' }
  // },

  async ({ event, step }) => {
    const { environmentId, runId } = event.data;

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
      }
    });
  },
);
