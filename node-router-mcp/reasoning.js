/**
 * Per-provider reasoning adapters.
 *
 * The two upstreams disagree completely on how "thinking" is expressed, and
 * neither matches OpenAI's documented shape:
 *
 * - Kilo: its published request schema omits any reasoning field, yet the
 *   gateway accepts `reasoning: {effort}` or `reasoning_effort` for models that
 *   advertise support via `supported_parameters` on GET /models. Support is
 *   discovered per model, then the field is chosen by advertisement. A 400
 *   naming an unknown field falls back once to no reasoning field at all.
 *
 * - NIM: `chat_template_kwargs: {"enable_thinking": true}` plus a top-level
 *   `reasoning_budget`. NIM's OpenAPI marks the body `additionalProperties:
 * false`, yet every NVIDIA example passes `chat_template_kwargs`, so the
 *   documented schema and the real API disagree; the examples are followed.
 *   `stream` defaults to TRUE there, so it is set false explicitly. Thinking
 *   comes back on `message.reasoning_content` and is stripped here.
 */
const KILO_REASONING_FIELDS = ["reasoning", "reasoning_effort"];
export function buildKiloRequest(opts) {
    const body = {
        model: opts.model,
        messages: opts.messages,
        stream: false,
        temperature: opts.temperature ?? 0.2,
    };
    if (opts.topP !== undefined)
        body.top_p = opts.topP;
    if (opts.maxTokens !== undefined)
        body.max_tokens = opts.maxTokens;
    if (opts.seed !== undefined)
        body.seed = opts.seed;
    if (opts.supportedParameters && opts.supportedParameters.has("reasoning")) {
        body.reasoning = { effort: opts.reasoning };
        opts.decision.applied = true;
        opts.decision.mechanism = "reasoning_object";
    }
    else if (opts.supportedParameters && opts.supportedParameters.has("reasoning_effort")) {
        body.reasoning_effort = opts.reasoning;
        opts.decision.applied = true;
        opts.decision.mechanism = "reasoning_effort";
    }
    else if (opts.supportedParameters === null) {
        // Catalog unreachable: try the OpenRouter-style field and let the 400
        // fallback handle rejection, rather than silently dropping reasoning.
        body.reasoning = { effort: opts.reasoning };
        opts.decision.applied = false;
        opts.decision.mechanism = "none";
        opts.decision.reason = "catalog unreachable; speculative reasoning field sent";
    }
    else {
        opts.decision.applied = false;
        opts.decision.mechanism = "none";
        opts.decision.reason = `model advertises no reasoning parameter (supported_parameters: ${[...opts.supportedParameters].sort().join(", ") || "empty"})`;
    }
    return {
        url: `${opts.baseUrl}/chat/completions`,
        headers: {
            "Content-Type": "application/json",
            ...(opts.apiKey ? { Authorization: `Bearer ${opts.apiKey}` } : {}),
        },
        body,
    };
}
export function kiloReasoningFields() {
    return KILO_REASONING_FIELDS;
}
export function stripKiloReasoningFields(body) {
    for (const field of KILO_REASONING_FIELDS)
        delete body[field];
}
/**
 * NIM reasoning shapes differ per model, and NIM does not publish which model
 * accepts which field - it only says no, with a 400, after the fact.

 * Verified live against `integrate.api.nvidia.com`:
 * - `stepfun-ai/step-5-preview` (Step 5): accepts `reasoning_effort`;
 *   REJECTS both `reasoning_budget` and `reasoning` with
 *   "Unsupported parameter(s)". It reasons unconditionally (600B MoE).
 * - `nvidia/nemotron-3.5-*` (Nemotron): accepts `reasoning_budget` plus
 *   `chat_template_kwargs.enable_thinking`, per NVIDIA's own examples.
 *
 * So the adapter sends the most expressive shape it can, and retries once
 * without whatever field the 400 names. That retry is what makes tier 1 usable
 * at all: without it, a permanently-rejected field would disable the tier.
 */
const NIM_CANDIDATE_FIELDS = [
    "reasoning_effort",
    "reasoning_budget",
    "chat_template_kwargs",
];
export function buildNimRequest(opts) {
    const reject = (field) => opts.rejectedFields?.has(field) ?? false;
    const body = {
        model: opts.model,
        messages: opts.messages,
        stream: false, // NIM defaults to true; explicit false is required
        temperature: opts.temperature ?? 0.2,
        top_p: opts.topP ?? 0.95,
        max_tokens: opts.maxTokens ?? 8192,
    };
    if (!reject("reasoning_effort"))
        body.reasoning_effort = opts.reasoning;
    if (!reject("reasoning_budget")) {
        body.reasoning_budget = opts.reasoningBudget;
        body.max_tokens = Math.max(body.max_tokens, opts.reasoningBudget);
    }
    if (!reject("chat_template_kwargs"))
        body.chat_template_kwargs = { enable_thinking: true };
    if (opts.seed !== undefined)
        body.seed = opts.seed;
    return {
        url: `${opts.baseUrl}/chat/completions`,
        headers: {
            "Content-Type": "application/json",
            ...(opts.apiKey ? { Authorization: `Bearer ${opts.apiKey}` } : {}),
        },
        body,
    };
}
/** Field names NIM names when rejecting an unsupported parameter. */
export function nimCandidateFields() {
    return NIM_CANDIDATE_FIELDS;
}
/**
 * Parse NIM's "Unsupported parameter(s): `a`, `b`" message into field names.
 */
export function parseUnsupportedParameters(message) {
    const match = /Unsupported parameter\(s\):\s*([^"]+)/i.exec(message);
    const captured = match?.[1];
    if (!captured)
        return [];
    return captured
        .split(",")
        .map((part) => part.trim().replace(/`/g, ""))
        .filter(Boolean);
}
/** Strip fields from a request body by name. */
export function stripNimFields(body, fields) {
    for (const field of fields)
        delete body[field];
}
/**
 * Providers return thinking in different fields, and both are non-OpenAI:
 *
 * - NIM: `message.reasoning_content` (and streaming `delta.reasoning_content`).
 * - Kilo: `message.reasoning` plus a parallel `message.reasoning_details` array,
 *   as seen on live responses from `kilo-auto/free` and the Nemotron tiers.
 *
 * A model that produces reasoning but no content has failed to answer at all,
 * which must be reported distinctly from "returned nothing" - otherwise a
 * degraded free tier is indistinguishable from a transport failure. `hadReasoning`
 * exists precisely so the caller can say which happened.
 */
const REASONING_FIELDS = ["reasoning_content", "reasoning", "reasoning_details"];
export function stripReasoningContent(raw) {
    if (!raw || typeof raw !== "object")
        return { content: "", hadReasoning: false };
    const root = raw;
    const choice = root.choices?.[0];
    const message = choice?.message ?? {};
    const content = typeof message.content === "string" ? message.content : "";
    const hadReasoning = REASONING_FIELDS.some((field) => {
        const value = message[field];
        if (typeof value === "string")
            return value.length > 0;
        return Array.isArray(value) && value.length > 0;
    });
    return { content, hadReasoning };
}
export function extractUsage(raw) {
    const usage = raw?.usage;
    const prompt = Number(usage?.prompt_tokens ?? 0) || 0;
    const completion = Number(usage?.completion_tokens ?? 0) || 0;
    return {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: Number(usage?.total_tokens ?? prompt + completion) || 0,
    };
}
