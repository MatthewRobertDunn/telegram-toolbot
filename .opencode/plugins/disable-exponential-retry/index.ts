import { Plugin } from "@opencode/plugin"

const DELAY_MS = 30_000

export default Plugin.define({
  id: "disable-exponential-retry",
  setup: async (ctx) => {
    await ctx.session.hook("retry", (event) => {
      if (event.error.type !== "provider.rate-limit") return

      // Keep OpenCode's "give up" decisions if it has already exhausted retries
      if (!event.decision.retry) return

      event.decision = { retry: true, delay: DELAY_MS }

    })
  },
})
