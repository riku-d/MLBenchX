/**
 * Top-level routing between the three screens.
 *
 * The route lives in the URL hash rather than in state alone, so a reload keeps you in
 * the workspace you were working in. That matters more here than in most apps: the
 * workspace holds edited code and a pipeline someone has spent time configuring, and
 * losing it to an accidental refresh would be the worst kind of data loss — the kind
 * the user caused and cannot undo.
 */

import { useCallback, useEffect, useState } from "react";
import { Landing } from "./landing/Landing";
import { UploadScreen } from "./workspace/UploadScreen";
import { Workspace } from "./workspace/Workspace";

type Route =
  | { screen: "landing" }
  | { screen: "upload" }
  | { screen: "workspace"; datasetId: string };

function parseHash(): Route {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("workspace/")) {
    const datasetId = hash.slice("workspace/".length);
    if (datasetId) return { screen: "workspace", datasetId };
  }
  if (hash === "upload") return { screen: "upload" };
  return { screen: "landing" };
}

function toHash(route: Route): string {
  if (route.screen === "workspace") return `#/workspace/${route.datasetId}`;
  if (route.screen === "upload") return "#/upload";
  return "#/";
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const go = useCallback((next: Route) => {
    window.location.hash = toHash(next);
    setRoute(next);
  }, []);

  if (route.screen === "workspace") {
    return (
      <Workspace
        key={route.datasetId}
        datasetId={route.datasetId}
        onExit={() => go({ screen: "upload" })}
      />
    );
  }

  if (route.screen === "upload") {
    return (
      <UploadScreen
        onReady={(dataset) => go({ screen: "workspace", datasetId: dataset.id })}
        onBack={() => go({ screen: "landing" })}
      />
    );
  }

  return <Landing onStart={() => go({ screen: "upload" })} />;
}
