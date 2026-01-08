import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: [
    'heuristically-unpreventive-dimple.ngrok-free.dev',
     'http://16.171.166.211',
    'https://16.171.166.211',
    // Add other ngrok domains here if needed
    // Pattern: '*.ngrok-free.dev' or '*.ngrok.io' (if supported)
  ],
};

export default nextConfig;
