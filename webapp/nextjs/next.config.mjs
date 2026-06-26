/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/spy500-breadth-backtest',
  assetPrefix: '/spy500-breadth-backtest',
  env: {
    NEXT_PUBLIC_BASE_PATH: '/spy500-breadth-backtest',
  },
}

export default nextConfig
