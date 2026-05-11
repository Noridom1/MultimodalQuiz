import { LoaderCircle } from "lucide-react";

function LoadingPanel() {
  return (
    <div className="loading-panel">
      <LoaderCircle className="spin" size={26} />
      <span>Loading workspace...</span>
    </div>
  );
}

export default LoadingPanel;
