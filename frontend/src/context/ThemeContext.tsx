import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

const THEME_KEY = "wcm_theme";
type Theme = "dark" | "light";

interface ThemeCtx {
  theme: Theme;
  isLight: boolean;
  setLight: (light: boolean) => void;
}

const Ctx = createContext<ThemeCtx | null>(null);

function applyTheme(theme: Theme) {
  if (theme === "light") document.body.classList.add("theme-light");
  else document.body.classList.remove("theme-light");
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(THEME_KEY) as Theme) || "dark"
  );

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setLight = (light: boolean) => {
    const t: Theme = light ? "light" : "dark";
    localStorage.setItem(THEME_KEY, t);
    setTheme(t);
  };

  return (
    <Ctx.Provider value={{ theme, isLight: theme === "light", setLight }}>
      {children}
    </Ctx.Provider>
  );
}

export function useTheme(): ThemeCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useTheme must be used within ThemeProvider");
  return c;
}
