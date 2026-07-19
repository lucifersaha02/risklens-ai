import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://lucifersaha02.github.io/risklens-ai/"),
  title: "RiskLens AI | Explainable Credit Risk Intelligence",
  description:
    "A documented data science platform for calibrated credit risk, SHAP explanations, responsible-AI diagnostics, monitoring and governed human review.",
  openGraph: {
    title: "RiskLens AI",
    description: "Credit risk, made accountable.",
    images: ["/risklens-ai/og.png"],
    type: "website",
  },
  twitter: { card: "summary_large_image", images: ["/risklens-ai/og.png"] },
  icons: { icon: "/risklens-ai/favicon.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
