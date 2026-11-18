/** Shared logger. Every module uses this so verbosity is changed in one place. */
export const logger = {
    info: (...args) => console.log(...args),
    warn: (...args) => console.warn(...args),
    error: (...args) => console.error(...args),
    debug: (...args) => {
        if (process.env.ROUTER_DEBUG)
            console.debug(...args);
    },
};
