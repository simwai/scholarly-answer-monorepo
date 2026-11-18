/**
 * Upstream providers and the ordered tier walk.
 *
 * A provider owns its model IDs. Model IDs are routed by publisher prefix, which
 * is the only reliable discriminator: `stepfun-ai/*` lives on NIM while
 * `stepfun/*` lives on Kilo, and those are *different models with different
 * prices* even though both are called Step 5 / Step 3.7.
 */
import { logger } from "./logging.js";
import { buildKiloRequest, buildNimRequest, extractUsage, kiloReasoningFields, parseUnsupportedParameters, stripNimFields, stripReasoningContent, } from "./reasoning.js";
function providerFor(modelId) {
    // stepfun-ai/ is NVIDIA NIM's namespace; nvidia/ and kilo-auto/ are Kilo's.
    return modelId.startsWith("stepfun-ai/") ? "nim" : "kilo";
}
/**
 * Sanitize lone surrogate characters (U+D800-U+DFFF) that would produce
 * invalid JSON escapes when serialized. Search result content scraped from
 * the web can contain these, and NIM's JSON parser rejects them.
 */
function sanitizeForJSON(value) {
    if (typeof value === "string") {
        return value.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
    }
    if (Array.isArray(value)) {
        return value.map(sanitizeForJSON);
    }
    if (value !== null && typeof value === "object") {
        const result = {};
        for (const [key, val] of Object.entries(value)) {
            result[key] = sanitizeForJSON(val);
        }
        return result;
    }
    return value;
}
async function fetchWithTimeout(url, init, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        // Sanitize lone surrogates in request body to prevent invalid JSON.
        // Web-scraped content can contain surrogate characters that JSON.stringify
        // encodes as \\udXXX, which NIM's parser rejects as "lone leading surrogate".
        if (typeof init.body === "string") {
            try {
                const parsed = JSON.parse(init.body);
                init.body = JSON.stringify(sanitizeForJSON(parsed));
            }
            catch {
                init.body = init.body.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
            }
        }
        return await fetch(url, { ...init, signal: controller.signal });
    }
    finally {
        clearTimeout(timer);
    }
}
function emptyCompletion(modelId, decisionReason) {
    return {
        ok: false,
        error: `${modelId} produced no content${decisionReason ? `; ${decisionReason}` : ""}`,
        errorType: "upstream_timeout",
    };
}
export class KiloProvider {
    config;
    kind = "kilo";
    catalogCache = null;
    constructor(config) {
        this.config = config;
    }
    owns(modelId) {
        return providerFor(modelId) === "kilo";
    }
    async supportedParameters(modelId) {
        const model = await this.catalogEntry(modelId);
        if (!model)
            return null;
        const params = model.supported_parameters;
        return Array.isArray(params) ? new Set(params.filter((p) => typeof p === "string")) : new Set();
    }
    async catalog() {
        if (this.catalogCache)
            return this.catalogCache;
        try {
            const response = await fetchWithTimeout(`${this.config.kiloBaseUrl}/models`, {
                method: "GET",
                headers: {
                    ...(this.config.kiloApiKey ? { Authorization: `Bearer ${this.config.kiloApiKey}` } : {}),
                },
            }, this.config.requestTimeoutMs);
            if (!response.ok)
                return null;
            const data = (await response.json());
            const map = new Map();
            for (const model of data.data ?? []) {
                const id = model.id;
                if (typeof id === "string")
                    map.set(id, model);
            }
            this.catalogCache = map;
            return map;
        }
        catch {
            return null;
        }
    }
    async catalogEntry(modelId) {
        const map = await this.catalog();
        return map?.get(modelId) ?? null;
    }
    async call(modelId, messages, temperature, topP, maxTokens, seed) {
        const supported = this.owns(modelId) ? await this.supportedParameters(modelId) : null;
        const decision = { applied: false, mechanism: "none", reason: undefined };
        let request = buildKiloRequest({
            baseUrl: this.config.kiloBaseUrl,
            apiKey: this.config.kiloApiKey,
            model: modelId,
            messages,
            temperature,
            topP,
            maxTokens,
            seed,
            reasoning: this.config.reasoning,
            supportedParameters: supported,
            decision,
        });
        const started = Date.now();
        const attempt = async (upstream) => (await fetchWithTimeout(upstream.url, { method: "POST", headers: upstream.headers, body: JSON.stringify(upstream.body) }, this.config.requestTimeoutMs));
        let response = await attempt(request);
        if (response.status === 400) {
            // Kilo's published schema omits the reasoning field; retry once without it.
            const stripped = { ...request.body };
            for (const field of kiloReasoningFields())
                delete stripped[field];
            request = { ...request, body: stripped };
            response = await attempt(request);
            if (response.ok) {
                decision.applied = false;
                decision.mechanism = "none";
                decision.reason = "upstream rejected the reasoning field; retried without it";
            }
        }
        return this.interpret(response, modelId, started, decision);
    }
    async interpret(response, modelId, started, decision) {
        if (!response.ok) {
            const body = await response.text().catch(() => "");
            return this.failure(response.status, body, decision);
        }
        const raw = await response.json().catch(() => undefined);
        const { content } = stripReasoningContent(raw);
        if (!content) {
            return emptyCompletion(modelId, decision.reason);
        }
        return { ok: true, content, modelUsed: raw?.model || modelId, upstreamSeconds: (Date.now() - started) / 1000, usage: extractUsage(raw) };
    }
    failure(status, body, decision) {
        const trimmed = body.slice(0, 300);
        // 400/404 mean the model ID or a field is wrong: configuration, not flakiness.
        if (status === 400 || status === 404) {
            return {
                ok: false,
                error: `${modelIdText(trimmed)} (${status})${decision.reason ? `; ${decision.reason}` : ""}`,
                errorType: "config_error",
                status,
            };
        }
        if (status === 429) {
            return { ok: false, error: `Kilo rate limited (429): ${trimmed}`, errorType: "upstream_rate_limited", status };
        }
        // Kilo maps an upstream 402 to 503 before the client sees it.
        if (status === 402 || status === 503) {
            return { ok: false, error: `Kilo quota/free-tier exhausted (${status}): ${trimmed}`, errorType: "upstream_rate_limited", status };
        }
        return { ok: false, error: `Kilo upstream error ${status}: ${trimmed}`, errorType: "upstream_timeout", status };
    }
}
function modelIdText(body) {
    return body.includes("does not exist") ? "model not available" : "upstream rejected request";
}
export class NimProvider {
    config;
    kind = "nim";
    /** Fields each model has rejected, so a bad field costs one 400 ever. */
    rejected = new Map();
    constructor(config) {
        this.config = config;
    }
    owns(modelId) {
        return providerFor(modelId) === "nim";
    }
    async supportedParameters() {
        // NIM's list endpoint returns no metadata, so reasoning support cannot be
        // discovered from it. The adapter learns from rejections instead.
        return null;
    }
    rejectedFor(modelId) {
        let set = this.rejected.get(modelId);
        if (!set) {
            set = new Set();
            this.rejected.set(modelId, set);
        }
        return set;
    }
    async call(modelId, messages, temperature, topP, maxTokens, seed) {
        if (!this.config.nimApiKey) {
            return { ok: false, error: `NIM tier ${modelId} needs NVIDIA_API_KEY`, errorType: "config_error" };
        }
        const request = buildNimRequest({
            baseUrl: this.config.nimBaseUrl,
            apiKey: this.config.nimApiKey,
            model: modelId,
            messages,
            temperature,
            topP,
            maxTokens,
            seed,
            reasoning: this.config.reasoning,
            reasoningBudget: this.config.nimReasoningBudget,
            rejectedFields: this.rejectedFor(modelId),
        });
        return this.send(request, modelId);
    }
    async send(request, modelId) {
        const started = Date.now();
        try {
            const response = await fetchWithTimeout(request.url, { method: "POST", headers: request.headers, body: JSON.stringify(request.body) }, this.config.requestTimeoutMs);
            if (response.status === 400) {
                return this.handleRejection(request, modelId, response);
            }
            if (response.status === 202) {
                // NIM returns 202 with the result pending. Its async status endpoint is
                // not documented in the public reference we could reach, so this is
                // reported rather than guessed at.
                return {
                    ok: false,
                    error: `NIM returned 202 (pending); the async status endpoint is undocumented, so the result cannot be retrieved - retry`,
                    errorType: "upstream_timeout",
                    status: 202,
                };
            }
            if (!response.ok) {
                const body = await response.text().catch(() => "");
                if (response.status === 404) {
                    return { ok: false, error: `NIM has no model ${modelId} (404)`, errorType: "config_error", status: 404 };
                }
                if (response.status === 401 || response.status === 403) {
                    return { ok: false, error: `NIM key rejected (${response.status})`, errorType: "config_error", status: response.status };
                }
                if (response.status === 429 || response.status === 503) {
                    return { ok: false, error: `NIM rate limited (${response.status})`, errorType: "upstream_rate_limited", status: response.status };
                }
                return {
                    ok: false,
                    error: `NIM upstream error ${response.status}: ${body.slice(0, 200)}`,
                    errorType: "upstream_timeout",
                    status: response.status,
                };
            }
            const raw = await response.json().catch(() => undefined);
            const { content } = stripReasoningContent(raw);
            if (!content) {
                return emptyCompletion(modelId);
            }
            return {
                ok: true,
                content,
                modelUsed: raw?.model || modelId,
                upstreamSeconds: (Date.now() - started) / 1000,
                usage: extractUsage(raw),
            };
        }
        catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            const isTimeout = /abort/i.test(message);
            return {
                ok: false,
                error: isTimeout ? `NIM request timed out: ${message}` : `NIM request failed: ${message}`,
                errorType: isTimeout ? "upstream_timeout" : "upstream_rate_limited",
            };
        }
    }
    /**
     * A 400 naming unsupported parameters is a *shape* mismatch, not a broken
     * tier. Learn the rejected fields and retry once without them. This is what
     * makes tier 1 work: Step 5 rejects `reasoning_budget`, so without this the
     * tier would fail forever and the chain would never reach Step 5 at all.
     */
    async handleRejection(request, modelId, response) {
        const body = await response.text().catch(() => "");
        const unsupported = parseUnsupportedParameters(body);
        if (!unsupported.length) {
            return {
                ok: false,
                error: `NIM rejected ${modelId} (400): ${body.slice(0, 300)}`,
                errorType: "config_error",
                status: 400,
            };
        }
        const known = this.rejectedFor(modelId);
        for (const field of unsupported)
            known.add(field);
        const stripped = { ...request.body };
        stripNimFields(stripped, unsupported);
        const retried = await fetchWithTimeout(request.url, { method: "POST", headers: request.headers, body: JSON.stringify(stripped) }, this.config.requestTimeoutMs);
        if (!retried.ok) {
            const retryBody = await retried.text().catch(() => "");
            if (retried.status === 400) {
                const more = parseUnsupportedParameters(retryBody);
                for (const field of more)
                    known.add(field);
            }
            return {
                ok: false,
                error: `NIM still rejects ${modelId} after dropping ${unsupported.join(", ")} (${retried.status}): ${retryBody.slice(0, 200)}`,
                errorType: "config_error",
                status: retried.status,
            };
        }
        const raw = await retried.json().catch(() => undefined);
        const { content } = stripReasoningContent(raw);
        if (!content) {
            return emptyCompletion(modelId);
        }
        logger.info("NIM %s accepted a reduced reasoning shape (dropped %s)", modelId, unsupported.join(", "));
        return {
            ok: true,
            content,
            modelUsed: raw?.model || modelId,
            upstreamSeconds: 0,
            usage: extractUsage(raw),
        };
    }
}
