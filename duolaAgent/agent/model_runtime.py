"""Resolve at admission; never mutate an admitted turn's runtime."""
from duolaAgent.llm.providers.factory import build_provider_snapshot


class ModelRuntimeResolver:
    def __init__(self, *, runtime=None, config=None, config_loader=None):
        self._runtime = runtime
        self._config = config.model_copy(deep=True) if config is not None else None
        self._config_loader = config_loader
        self._invalidated = False
        self._snapshots = {}
        self._retired = []

    @property
    def current(self):
        if self._runtime is None:
            self._runtime = self.admit()
        return self._runtime

    def invalidate(self):
        self._invalidated = True

    def refresh(self):
        if self._config_loader:
            self._config = self._config_loader().model_copy(deep=True)
        self._retired.extend(self._snapshots.values())
        self._snapshots.clear()
        self._invalidated = False

    def admit(self, session=None, *, model=None, preset=None):
        if self._invalidated:
            self.refresh()
        selected = preset or (session.metadata.get("model_preset") if session else None)
        if self._config is None:
            if selected:
                raise ValueError("Model presets require a configuration")
            return self._runtime.with_overrides(model=model) if model else self._runtime
        key = (selected, model)
        if key not in self._snapshots:
            self._snapshots[key] = build_provider_snapshot(self._config, preset=selected, model=model)
        runtime = self._snapshots[key].runtime
        if key == (None, None):
            self._runtime = runtime
        return runtime

    async def aclose(self):
        seen = set()
        runtimes = [s.runtime for s in [*self._snapshots.values(), *self._retired]]
        if self._runtime:
            runtimes.append(self._runtime)
        for runtime in runtimes:
            if id(runtime.provider) not in seen:
                seen.add(id(runtime.provider))
                await runtime.provider.aclose()
