from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class AdmetModel(Protocol):
    endpoint: str

    def predict(self, smiles: str) -> float:
        ...


@dataclass
class MockAdmetModel:
    endpoint: str

    def predict(self, smiles: str) -> float:
        # Deterministic pseudo-score in [0,1] for pipeline continuity.
        h = hashlib.sha256(f"{self.endpoint}|{smiles}".encode("utf-8")).hexdigest()
        return (int(h[:8], 16) % 10_000) / 10_000.0


class ModelRegistry:
    """Registry enabling multiple ADMET endpoints and future model backends."""

    def __init__(self) -> None:
        self._models: dict[str, AdmetModel] = {}

    def register(self, endpoint: str, model: AdmetModel) -> None:
        self._models[endpoint] = model

    def get(self, endpoint: str) -> AdmetModel:
        if endpoint not in self._models:
            raise KeyError(f"ADMET endpoint not registered: {endpoint}")
        return self._models[endpoint]


def build_model_registry(endpoints: list[str], use_mock_models: bool = True) -> ModelRegistry:
    registry = ModelRegistry()
    for endpoint in endpoints:
        if use_mock_models:
            registry.register(endpoint, MockAdmetModel(endpoint=endpoint))
        else:
            # In v1 this path still uses mock models unless real integration is supplied.
            registry.register(endpoint, MockAdmetModel(endpoint=endpoint))
    return registry


def predict_admet(df: pd.DataFrame, registry: ModelRegistry, endpoints: list[str]) -> pd.DataFrame:
    out = df.copy()
    for endpoint in endpoints:
        model = registry.get(endpoint)
        out[f"admet_{endpoint}"] = out["canonical_smiles"].astype(str).map(model.predict)
    return out

