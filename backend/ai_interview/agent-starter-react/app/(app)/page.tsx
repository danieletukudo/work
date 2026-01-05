import { Suspense } from 'react';
import { headers } from 'next/headers';
import { App } from '@/components/app/app';
import { getAppConfig } from '@/lib/utils';

export default async function Page() {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);

  return (
    <Suspense>
      <App appConfig={appConfig} />
    </Suspense>
  );
}
