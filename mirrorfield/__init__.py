"""
Mirrorfield AI — Geometric Safety Framework

A rigorous framework for AI safety evaluation using geometric features
in embedding spaces.

Core Components:
- geometry: Native k-NN geometric features (+6.4% validated improvement)
- api: Geometric state monitoring for real-time safety diagnostics
- mcp: MCP server integration for uncertainty reporting

Usage:
    from mirrorfield.geometry import GeometryBundle

    bundle = GeometryBundle(reference_embeddings, k=50)
    results = bundle.compute(query_embeddings)
"""

__version__ = "2.0.0"
__author__ = "Mirrorfield Project"
