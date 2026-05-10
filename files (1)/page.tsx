"use client";

import { useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Product {
  product_id: string;
  name: string;
  description: string;
  category: string;
  manufacturer: string;
  certifications: string[];
  unit: string;
  score: number;
}

interface SearchResponse {
  query: string;
  results: Product[];
  retriever: string;
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const EXAMPLE_QUERIES = [
  "sterile latex-free examination gloves size M EN455",
  "Einmalspritze 10ml Luer-Lock",
  "wound dressing 10x10cm silicone sterile",
  "PTFE 4-0 CV surgical suture",
];

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [retriever, setRetriever] = useState<"bm25" | "dense" | "hybrid">("hybrid");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(
        `/api/search?q=${encodeURIComponent(query)}&retriever=${retriever}`
      );
      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data: SearchResponse = await res.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">
          Medical Procurement Search
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Retrieval benchmark demo — compare BM25, dense, and hybrid matching
        </p>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Search bar */}
        <div className="flex gap-3 mb-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Describe the product you need..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="bg-blue-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium
                       hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {/* Retriever selector */}
        <div className="flex gap-2 mb-6">
          {(["bm25", "dense", "hybrid"] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRetriever(r)}
              className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                retriever === r
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"
              }`}
            >
              {r === "bm25" ? "BM25 (keyword)" : r === "dense" ? "Dense (semantic)" : "Hybrid (RRF)"}
            </button>
          ))}
        </div>

        {/* Example queries */}
        <div className="mb-6">
          <p className="text-xs text-gray-500 mb-2">Try these examples:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => setQuery(q)}
                className="text-xs text-blue-600 bg-blue-50 px-3 py-1 rounded-full
                           hover:bg-blue-100 transition-colors border border-blue-200"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm mb-4">
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm text-gray-500">
                {results.results.length} results · {results.latency_ms}ms · {results.retriever}
              </p>
            </div>

            <div className="space-y-3">
              {results.results.map((product, idx) => (
                <div
                  key={product.product_id}
                  className="bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-gray-400 font-mono">#{idx + 1}</span>
                        <h3 className="text-sm font-medium text-gray-900 truncate">
                          {product.name}
                        </h3>
                      </div>
                      <p className="text-xs text-gray-500 mb-2 line-clamp-2">
                        {product.description}
                      </p>
                      <div className="flex flex-wrap gap-1.5 items-center">
                        <span className="text-xs text-gray-500">{product.manufacturer}</span>
                        <span className="text-gray-300">·</span>
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                          {product.category}
                        </span>
                        {product.certifications.map((cert) => (
                          <span
                            key={cert}
                            className="text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full border border-green-200"
                          >
                            {cert}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-sm font-semibold text-blue-600">
                        {(product.score * 100).toFixed(1)}
                      </div>
                      <div className="text-xs text-gray-400">score</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
