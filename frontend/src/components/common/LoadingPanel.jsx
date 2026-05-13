import { LoaderCircle } from "lucide-react";

function LoadingPanel({ label = "Loading workspace..." }) {
  return (
    <div className="loading-panel">
      <LoaderCircle className="spin" size={26} />
      <span>{label}</span>
    </div>
  );
}

export default LoadingPanel;
