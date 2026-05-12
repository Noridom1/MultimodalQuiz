import { useEffect, useState } from "react";
import { ArrowLeft, LoaderCircle, Search } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import VisionQBrandLink from "../components/common/VisionQBrandLink";
import UserAccountMenu from "../components/common/UserAccountMenu";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { heroStyle } from "../utils/ui";

function SearchPage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialQuery = searchParams.get("q") || "";
  const [query, setQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  useEffect(() => {
    setQuery(initialQuery);
    setDebouncedQuery(initialQuery);
  }, [initialQuery]);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      const next = query.trim();
      setDebouncedQuery(next);
      setSearchParams(next ? { q: next } : {}, { replace: true });
    }, 250);
    return () => window.clearTimeout(timeoutId);
  }, [query, setSearchParams]);

  useEffect(() => {
    if (!debouncedQuery) {
      setResults([]);
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    api
      .listNotebooks(debouncedQuery)
      .then((data) => {
        if (active) setResults(data);
      })
      .catch(() => {
        if (active) setResults([]);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [debouncedQuery, user?.id]);

  return (
    <div className="page-shell search-page">
      <header className="topbar">
        <VisionQBrandLink />
        <div className="topbar-actions">
          {user ? <UserAccountMenu /> : null}
        </div>
      </header>

      <section className="search-screen-head">
        <button className="ghost-pill" onClick={() => navigate("/")}>
          <ArrowLeft size={18} />
          Back
        </button>
        <label className="search-screen-input" aria-label="Search notebooks">
          <Search size={18} />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by notebook title or description"
            autoFocus
          />
        </label>
      </section>

      <section className="search-screen-results">
        {!debouncedQuery ? (
          <p className="search-screen-hint">Type to search notebooks.</p>
        ) : loading ? (
          <div className="search-screen-loading">
            <LoaderCircle className="spin" size={18} />
            <span>Searching notebooks...</span>
          </div>
        ) : results.length === 0 ? (
          <p className="search-screen-hint">No notebooks found for "{debouncedQuery}".</p>
        ) : (
          <div className="search-results-list">
            {results.map((notebook, index) => (
              <button
                key={notebook.id}
                className="search-result-item"
                onClick={() => navigate(`/notebooks/${notebook.id}`)}
              >
                <div className="search-result-thumb" style={heroStyle(notebook, index)} />
                <div className="search-result-body">
                  <h3>{notebook.title}</h3>
                  <p>
                    {notebook.question_count || 0} questions - {notebook.source_count || 0} sources
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default SearchPage;
