import type React from "react";
import type { SVGProps } from "react";

type LogoProps = SVGProps<SVGSVGElement> & { size?: number };

function defaults(props: LogoProps, size = 40) {
  const s = props.size ?? size;
  return { width: s, height: s, viewBox: props.viewBox, ...props, size: undefined };
}

/* ── OVH ─────────────────────────────────────────────────────────────────── */

export function OvhLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      <path
        d="M50 8L10 85h25l10-25h10l-10 25h25L50 8Z"
        fill="#000E9C"
      />
      <path
        d="M55 60h15l20-52H70L55 60Z"
        fill="#59D3F4"
      />
    </svg>
  );
}

/* ── AWS ─────────────────────────────────────────────────────────────────── */

export function AwsLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      {/* Cloud shape */}
      <path
        d="M30 65c-2 0-5-1-7-3s-3-5-3-8c0-4 2-7 5-9-1-2-1-4-1-5 0-4 1-7 4-10s6-4 10-4c5 0 9 2 12 6 2-1 4-2 6-2 4 0 7 1 9 4s4 6 4 9v1c4 1 7 3 9 6s3 6 3 9c0 2-1 4-2 6H30Z"
        fill="#FF9900"
      />
      {/* Smile arrow */}
      <path
        d="M25 75c8 5 18 7 28 5s18-8 24-16"
        stroke="#FF9900"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
      <path d="M72 67l5-3-1 5" fill="#FF9900" />
    </svg>
  );
}

/* ── GCP ─────────────────────────────────────────────────────────────────── */

export function GcpLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      {/* Hexagon */}
      <path d="M50 10L87 30v40L50 90 13 70V30L50 10Z" fill="#4285F4" />
      <path d="M50 10L87 30v40L50 90" fill="#34A853" />
      <path d="M50 90L13 70V30" fill="#FBBC04" />
      <path d="M50 10L13 30" fill="#EA4335" />
      {/* Inner shape */}
      <circle cx="50" cy="50" r="16" fill="white" opacity="0.9" />
      <circle cx="50" cy="50" r="10" fill="#4285F4" />
    </svg>
  );
}

/* ── Scaleway ────────────────────────────────────────────────────────────── */

export function ScalewayLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      <rect x="15" y="15" width="70" height="70" rx="14" fill="#4F0599" />
      <path
        d="M35 60V45c0-3 1-5 3-7s5-3 8-3h8c3 0 5 1 7 3l-5 5c-1-1-2-1-3-1h-6c-1 0-2 0-2 1s-1 2-1 3v12h10l-5 7H35Z"
        fill="white"
      />
    </svg>
  );
}

/* ── Infomaniak ─────────────────────────────────────────────────────────── */

export function InfomaniakLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      {/* Swiss cross on blue background */}
      <rect x="10" y="10" width="80" height="80" rx="16" fill="#0098FF" />
      <rect x="44" y="25" width="12" height="50" rx="3" fill="white" />
      <rect x="25" y="44" width="50" height="12" rx="3" fill="white" />
    </svg>
  );
}

/* ── Azure ──────────────────────────────────────────────────────────────── */

export function AzureLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      <path d="M25 80L50 15l15 30H40l35 35H25Z" fill="#0078D4" />
      <path d="M55 45l20-10v45L55 45Z" fill="#50E6FF" opacity="0.7" />
    </svg>
  );
}

/* ── Vast.ai ────────────────────────────────────────────────────────────── */

export function VastaiLogo(props: LogoProps) {
  return (
    <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
      <rect x="10" y="10" width="80" height="80" rx="14" fill="#1A1A2E" />
      {/* V shape */}
      <path
        d="M30 30l20 40 20-40"
        stroke="#00D4AA"
        strokeWidth="8"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      {/* GPU dots */}
      <circle cx="35" cy="75" r="3" fill="#00D4AA" />
      <circle cx="50" cy="75" r="3" fill="#00D4AA" />
      <circle cx="65" cy="75" r="3" fill="#00D4AA" />
    </svg>
  );
}

/* ── Lookup ───────────────────────────────────────────────────────────────── */

const LOGOS: Record<string, (props: LogoProps) => React.JSX.Element> = {
  ovh: OvhLogo,
  infomaniak: InfomaniakLogo,
  aws: AwsLogo,
  gcp: GcpLogo,
  azure: AzureLogo,
  scaleway: ScalewayLogo,
  vastai: VastaiLogo,
};

export function ProviderLogo({
  providerId,
  ...props
}: LogoProps & { providerId: string }) {
  const Logo = LOGOS[providerId];
  if (!Logo) {
    // Fallback: generic cloud icon
    return (
      <svg {...defaults(props)} viewBox="0 0 100 100" fill="none">
        <path
          d="M30 65c-2 0-5-1-7-3s-3-5-3-8c0-4 2-7 5-9-1-2-1-4-1-5 0-4 1-7 4-10s6-4 10-4c5 0 9 2 12 6 2-1 4-2 6-2 4 0 7 1 9 4s4 6 4 9v1c4 1 7 3 9 6s3 6 3 9c0 2-1 4-2 6H30Z"
          fill="currentColor"
          opacity="0.5"
        />
      </svg>
    );
  }
  return <Logo {...props} />;
}
