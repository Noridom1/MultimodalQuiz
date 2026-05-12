import { Link } from "react-router-dom";

/** Home link with VisionQ logo + wordmark (shared header branding). */
export default function VisionQBrandLink({ to = "/", ...props }) {
  return (
    <Link className="visionq-brand" to={to} {...props}>
      <span className="visionq-logo-mark" aria-hidden>
        VQ
      </span>
      <span className="visionq-wordmark">VisionQ</span>
    </Link>
  );
}
