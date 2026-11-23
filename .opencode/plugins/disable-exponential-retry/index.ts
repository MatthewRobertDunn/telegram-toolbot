import { Plugin } from "@opencode/plugin"

const DELAY_MS = 30_000 // Your fixed delay

export default Plugin.define({
  id: "disable-exponential-retry",
  setup: async (ctx) => {
    await ctx.session.hook("retry", (event) => {
      // Only modify provider rate limits (429 errors)
      if (event.error.type !== "provider.rate-limit") return

      // Keep OpenCode's "give up" decisions if it has already exhausted retries
      if (!event.decision.retry) return

      // Override to a fixed interval delay
      event.decision = { retry: true, delay: DELAY_MS }

      // Alternatively, to DISABLE retries completely, uncomment the line below:
      // event.decision = { retry: false }
    })
  },
})
