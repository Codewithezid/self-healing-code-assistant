window.APP_CONFIG = window.APP_CONFIG || {
  defaultProvider: "mistral",
  allowedProviders: ["mistral"],
  authRequired: false,
  maxIterationsCap: 3,
  validationTimeoutCap: 5,
  ragAvailable: true,
  ragDefaultEnabled: false,
  correctiveRagModes: ["fast", "balanced", "aggressive"],
  correctiveRagDefaultMode: "balanced",
  runtimeProfiles: ["custom", "fast", "balanced", "accurate"],
  defaultRuntimeProfile: "custom"
};
