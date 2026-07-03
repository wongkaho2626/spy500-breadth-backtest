/** @type {import('next').NextConfig} */
// GitHub Pages serves the site under /spy500-breadth-backtest, but local dev
// (and any non-CI build) should serve at the root. GitHub Actions always sets
// GITHUB_ACTIONS=true, so the Pages deploy keeps the base path automatically.
const isCI = process.env.GITHUB_ACTIONS === 'true'
const BASE_PATH = isCI ? '/spy500-breadth-backtest' : ''

const nextConfig = {
  output: 'export',
  basePath: BASE_PATH,
  assetPrefix: BASE_PATH,
  env: {
    NEXT_PUBLIC_BASE_PATH: BASE_PATH,
  },
}

export default nextConfig
