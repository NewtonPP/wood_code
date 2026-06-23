import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { User } from "../types";
import {
  checkSession,
  loginRequest,
  logoutRequest,
  signupRequest,
} from "../lib/api";

type AuthResult = { ok: boolean; error?: string };

interface AuthCtx {
  user: User | null;
  ready: boolean; // initial session check complete
  login: (email: string, password: string) => Promise<AuthResult>;
  signup: (email: string, password: string, displayName: string) => Promise<AuthResult>;
  logout: () => Promise<void>;
}

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      const u = await checkSession();
      if (u) setUser(u);
      setReady(true);
    })();
  }, []);

  // Shared handler for endpoints that return { ok, user } and auto-login.
  const handleAuthResponse = async (res: Response): Promise<AuthResult> => {
    if (!res.ok) {
      let error = "";
      try {
        const body = await res.json();
        error = body?.detail || "";
      } catch {
        error = await res.text();
      }
      return { ok: false, error };
    }
    const data = await res.json();
    setUser(data.user as User);
    return { ok: true };
  };

  const login: AuthCtx["login"] = async (email, password) => {
    try {
      return await handleAuthResponse(await loginRequest(email, password));
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  };

  const signup: AuthCtx["signup"] = async (email, password, displayName) => {
    try {
      return await handleAuthResponse(await signupRequest(email, password, displayName));
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  };

  const logout = async () => {
    await logoutRequest();
    setUser(null);
  };

  return (
    <Ctx.Provider value={{ user, ready, login, signup, logout }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAuth(): AuthCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth must be used within AuthProvider");
  return c;
}
