import { Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import NotebookPage from "./pages/NotebookPage";
import SignUpPage from "./pages/SignUpPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route path="/notebooks/:notebookId" element={<NotebookPage />} />
    </Routes>
  );
}

export default App;
