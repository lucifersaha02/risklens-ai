import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/risklens-ai",
  assetPrefix: "/risklens-ai/",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
