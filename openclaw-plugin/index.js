import { MemChipClient } from "./lib/client.js";
import { registerTools } from "./lib/tools.js";
import { registerHooks } from "./lib/hooks.js";

export default {
  id: "openclaw-memchip",
  name: "MemChip Memory",
  kind: "memory",

  configSchema: {
    jsonSchema: {
      type: "object",
      properties: {
        apiUrl:       { type: "string", default: "http://76.13.23.55/v1" },
        apiKey:       { type: "string" },
        userId:       { type: "string", default: "seiji" },
        agentId:      { type: "string", default: "lyn" },
        orgId:        { type: "string", default: "team-seiji" },
        autoRecall:   { type: "boolean", default: true },
        autoCapture:  { type: "boolean", default: true },
        topK:         { type: "number", default: 5 },
        sharedPools:  { type: "array", items: { type: "string" }, default: ["team"] },
        compactionFlush: { type: "boolean", default: true },
      },
      required: ["apiKey"],
    },
    uiHints: {
      apiKey: { label: "API Key", sensitive: true },
      apiUrl: { label: "API URL" },
    },
  },

  async activate(api) {
    const config = api.pluginConfig || {};
    const client = new MemChipClient({
      apiUrl:  config.apiUrl  || "http://76.13.23.55/v1",
      apiKey:  config.apiKey,
      userId:  config.userId  || "seiji",
      agentId: config.agentId || "lyn",
      orgId:   config.orgId   || "team-seiji",
    });

    registerTools(api, client);
    registerHooks(api, client, config);

    api.logger.info(`MemChip plugin activated for user: ${config.userId}`);
  },
};
