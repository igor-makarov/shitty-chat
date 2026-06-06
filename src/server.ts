import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  jsonSchema,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";
import { OpenAPIToolGenerator, type McpOpenAPITool } from "mcp-from-openapi";

// ── OpenAPI tool generation helpers ──────────────────────────────────

const OPENAPI_SPECS = [
  {
    name: "bus",
    url: "https://open-bus-stride-api.hasadna.org.il/openapi.json"
  },
  {
    name: "hebcal",
    url: "https://www.hebcal.com/api-docs/openapi.json",
    onlyOperations: ["getZmanim"]
  }
];

function mcpToolToAiTool(t: McpOpenAPITool) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (tool as any)({
    description:
      t.metadata.operationSummary ||
      t.metadata.operationDescription ||
      t.description ||
      `${t.metadata.method.toUpperCase()} ${t.metadata.path}`,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    inputSchema: jsonSchema(t.inputSchema as any),
    execute: async (input: Record<string, unknown>) => {
      let path = t.metadata.path;
      const query = new URLSearchParams();
      const headers: Record<string, string> = {};
      let bodyParts: Array<{ key: string; value: unknown }> = [];

      for (const m of t.mapper) {
        const value = input[m.inputKey];
        if (value === undefined || value === null) continue;

        switch (m.type) {
          case "path":
            path = path.replace(
              `{${m.key}}`,
              encodeURIComponent(String(value))
            );
            break;
          case "query":
            if (Array.isArray(value)) {
              for (const v of value) query.append(m.key, String(v));
            } else {
              query.set(m.key, String(value));
            }
            break;
          case "header":
            headers[m.key] = String(value);
            break;
          case "body":
            bodyParts.push({ key: m.key, value });
            break;
          case "cookie":
            headers["cookie"] =
              (headers["cookie"] || "") +
              `${m.key}=${encodeURIComponent(String(value))}; `;
            break;
        }
      }

      const baseUrl =
        t.metadata.servers?.[0]?.url ??
        "https://open-bus-stride-api.hasadna.org.il";
      const qs = query.toString();
      const url = `${baseUrl}${path}${qs ? "?" + qs : ""}`;

      // Build body object from assembled parts
      let body: Record<string, unknown> | undefined;
      if (bodyParts.length > 0) {
        body = {};
        for (const { key, value } of bodyParts) {
          body[key] = value;
        }
      }

      const response = await fetch(url, {
        method: t.metadata.method.toUpperCase(),
        headers: {
          accept: "application/json",
          ...headers
        },
        body: body ? JSON.stringify(body) : undefined
      });

      if (!response.ok) {
        return {
          error: true,
          status: response.status,
          statusText: response.statusText,
          body: await response.text().catch(() => "")
        };
      }

      return await response.json();
    }
  });
}

// ── Agent ────────────────────────────────────────────────────────────

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;
  private _openapiToolsPromise?: Promise<Record<string, unknown>>;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  private async getOpenapiTools() {
    if (!this._openapiToolsPromise) {
      this._openapiToolsPromise = this._loadOpenapiTools();
    }
    return this._openapiToolsPromise;
  }

  private async _loadOpenapiTools() {
    const allTools: Record<string, unknown> = {};

    for (const specDef of OPENAPI_SPECS) {
      console.log(`Loading OpenAPI spec [${specDef.name}] from`, specDef.url);
      const res = await fetch(specDef.url);
      if (!res.ok)
        throw new Error(
          `Failed to fetch OpenAPI spec [${specDef.name}]: ${res.status} ${res.statusText}`
        );
      const spec = (await res.json()) as object;

      const generator = await OpenAPIToolGenerator.fromJSON(spec);
      const generatorOptions: Parameters<typeof generator.generateTools>[0] =
        {};

      if (specDef.onlyOperations) {
        generatorOptions.filterFn = (op: { operationId?: string }) =>
          op.operationId != null &&
          specDef.onlyOperations!.includes(op.operationId);
      }

      const openapiTools = await generator.generateTools(generatorOptions);
      console.log(
        `Generated ${openapiTools.length} tools from [${specDef.name}]`
      );

      for (const ot of openapiTools) {
        const name = ot.name.replace(/[^a-zA-Z0-9_-]/g, "_");
        allTools[name] = mcpToolToAiTool(ot);
      }
    }

    return allTools;
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const openapiTools = await this.getOpenapiTools();
    const opencode = createOpenAICompatible({
      name: "opencode",
      baseURL: "https://opencode.ai/zen/go/v1",
      headers: {
        Authorization: `Bearer ${this.env.OPENCODE_API_KEY}`
      }
    });

    const result = streamText({
      model: opencode("deepseek-v4-flash"),
      system: `You are a helpful assistant that can call tools. Use the tools in a loop before answering user questions.`,
      messages: await convertToModelMessages(this.messages),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Tools generated from OpenAPI spec (public transit API)
        ...openapiTools,

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(30),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
