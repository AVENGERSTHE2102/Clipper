import { NextResponse } from "next/server";

import { createProxyResponse, fetchBackend } from "@/server/backend-api";
import { getServerSession } from "@/server/session";

export async function GET() {
  const session = await getServerSession();
  const localAppAccess = process.env.LOCAL_APP_ACCESS === "true";
  if (!session?.user?.id && !localAppAccess) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const upstream = await fetchBackend("/tasks/", {
    method: "GET",
    ...(session?.user?.id ? { userId: session.user.id } : {}),
    cache: "no-store",
  });

  return createProxyResponse(upstream);
}
