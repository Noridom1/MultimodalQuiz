import { Route, Routes } from "react-router-dom";
import RequireAuth from "./components/auth/RequireAuth";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import NotebookPage from "./pages/NotebookPage";
import SearchPage from "./pages/SearchPage";
import SavedPage from "./pages/SavedPage";
import SignUpPage from "./pages/SignUpPage";

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route element={<RequireAuth />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/saved" element={<SavedPage />} />
        <Route path="/saved/:listId" element={<SavedPage />} />
        <Route path="/notebooks/:notebookId" element={<NotebookPage />} />
      </Route>
    </Routes>
  );
}

export default App;
