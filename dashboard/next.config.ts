import type { NextConfig } from "next";

const apiPort = process.env.NAVBUDDY_API_PORT || "8765";

const config: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `http://localhost:${apiPort}/api/:path*`,
      },
    ];
  },
};

export default config;
