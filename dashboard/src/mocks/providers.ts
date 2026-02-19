import type {
  ConnectionTestResult,
  ProviderComputeInstance,
  ProviderDetail,
  ProviderRegion,
  ProviderStorageOption,
  ProviderSummary,
} from "@/types/api";

/* ── Provider summaries (list page) ──────────────────────────────────────── */

export const MOCK_PROVIDERS: ProviderSummary[] = [
  {
    id: "ovh",
    display_name: "OVH Public Cloud",
    description: "European cloud provider based on OpenStack",
    enabled: true,
    status: "connected",
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage via OpenStack Swift",
      },
      {
        type: "compute",
        name: "Public Cloud Instances",
        description: "CPU and GPU compute instances via OpenStack Nova",
      },
    ],
  },
  {
    id: "infomaniak",
    display_name: "Infomaniak Public Cloud",
    description: "Swiss cloud provider based on OpenStack",
    enabled: false,
    status: "unconfigured",
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage via OpenStack Swift",
      },
      {
        type: "compute",
        name: "Public Cloud Instances",
        description: "CPU and GPU compute instances via OpenStack Nova",
      },
    ],
  },
  {
    id: "aws",
    display_name: "Amazon Web Services",
    description: "Global cloud platform — S3, EC2, and more",
    enabled: false,
    status: "unconfigured",
    capabilities: [
      {
        type: "storage",
        name: "Amazon S3",
        description: "Scalable object storage (Simple Storage Service)",
      },
      {
        type: "compute",
        name: "Amazon EC2",
        description: "Elastic Compute Cloud — CPU and GPU instances",
      },
    ],
  },
  {
    id: "gcp",
    display_name: "Google Cloud Platform",
    description: "Global cloud platform — GCS, Compute Engine, and more",
    enabled: false,
    status: "error",
    capabilities: [
      {
        type: "storage",
        name: "Cloud Storage",
        description: "Google Cloud Storage buckets",
      },
      {
        type: "compute",
        name: "Compute Engine",
        description: "CPU and GPU virtual machines",
      },
    ],
  },
  {
    id: "azure",
    display_name: "Microsoft Azure",
    description: "Global cloud platform — Blob Storage, Virtual Machines, and more",
    enabled: false,
    status: "unconfigured",
    capabilities: [
      {
        type: "storage",
        name: "Blob Storage",
        description: "Azure Blob Storage containers",
      },
      {
        type: "compute",
        name: "Virtual Machines",
        description: "CPU and GPU virtual machines",
      },
    ],
  },
  {
    id: "scaleway",
    display_name: "Scaleway",
    description: "European cloud provider with GPU instances and S3 storage",
    enabled: false,
    status: "configured",
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage (Scaleway Object Storage)",
      },
      {
        type: "compute",
        name: "Instances",
        description: "CPU and GPU compute instances (Scaleway Instances)",
      },
    ],
  },
  {
    id: "vastai",
    display_name: "Vast.ai",
    description: "GPU compute marketplace for deep learning",
    enabled: false,
    status: "unconfigured",
    capabilities: [
      {
        type: "compute",
        name: "GPU Instances",
        description: "On-demand and interruptible GPU instances",
      },
    ],
  },
];

/* ── Provider details ────────────────────────────────────────────────────── */

export const MOCK_PROVIDER_DETAILS: Record<string, ProviderDetail> = {
  ovh: {
    id: "ovh",
    display_name: "OVH Public Cloud",
    description: "European cloud provider based on OpenStack",
    website: "https://www.ovhcloud.com/en/public-cloud/",
    connector_type: "openstack",
    enabled: true,
    status: "connected",
    status_message: "Connected — 42 flavors available",
    has_credentials: true,
    config: { region: "GRA11", user_domain_name: "Default" },
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage via OpenStack Swift",
      },
      {
        type: "compute",
        name: "Public Cloud Instances",
        description: "CPU and GPU compute instances via OpenStack Nova",
      },
    ],
    credential_fields: [
      {
        key: "auth_url",
        label: "Keystone Auth URL",
        required: true,
        secret: false,
        placeholder: "https://auth.cloud.ovh.net/v3",
      },
      {
        key: "username",
        label: "OpenStack Username",
        required: true,
        secret: false,
        placeholder: "",
      },
      {
        key: "password",
        label: "OpenStack Password",
        required: true,
        secret: true,
        placeholder: "",
      },
      {
        key: "project_id",
        label: "Project ID",
        required: true,
        secret: false,
        placeholder: "",
      },
    ],
    config_fields: [
      {
        key: "region",
        label: "Default Region",
        default: "GRA11",
        description: "",
      },
      {
        key: "user_domain_name",
        label: "User Domain Name",
        default: "Default",
        description: "OpenStack user domain (usually 'Default')",
      },
      {
        key: "project_domain_name",
        label: "Project Domain Name",
        default: "Default",
        description: "OpenStack project domain (usually 'Default')",
      },
    ],
    last_checked_at: "2026-02-19T10:30:00Z",
    created_at: "2026-02-18T09:00:00Z",
    updated_at: "2026-02-19T10:30:00Z",
  },

  infomaniak: {
    id: "infomaniak",
    display_name: "Infomaniak Public Cloud",
    description: "Swiss cloud provider based on OpenStack",
    website: "https://www.infomaniak.com/en/hosting/public-cloud",
    connector_type: "openstack",
    enabled: false,
    status: "unconfigured",
    status_message: "",
    has_credentials: false,
    config: {},
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage via OpenStack Swift",
      },
      {
        type: "compute",
        name: "Public Cloud Instances",
        description: "CPU and GPU compute instances via OpenStack Nova",
      },
    ],
    credential_fields: [
      {
        key: "auth_url",
        label: "Keystone Auth URL",
        required: true,
        secret: false,
        placeholder: "https://api.pub1.infomaniak.cloud/identity/v3",
      },
      {
        key: "username",
        label: "OpenStack Username",
        required: true,
        secret: false,
        placeholder: "",
      },
      {
        key: "password",
        label: "OpenStack Password",
        required: true,
        secret: true,
        placeholder: "",
      },
      {
        key: "project_id",
        label: "Project ID",
        required: true,
        secret: false,
        placeholder: "",
      },
    ],
    config_fields: [
      {
        key: "region",
        label: "Default Region",
        default: "dc3-a",
        description: "",
      },
      {
        key: "user_domain_name",
        label: "User Domain Name",
        default: "Default",
        description: "OpenStack user domain (usually 'Default')",
      },
      {
        key: "project_domain_name",
        label: "Project Domain Name",
        default: "Default",
        description: "OpenStack project domain (usually 'Default')",
      },
    ],
    last_checked_at: null,
    created_at: null,
    updated_at: null,
  },

  aws: {
    id: "aws",
    display_name: "Amazon Web Services",
    description: "Global cloud platform — S3, EC2, and more",
    website: "https://aws.amazon.com/",
    connector_type: "aws",
    enabled: false,
    status: "unconfigured",
    status_message: "",
    has_credentials: false,
    config: {},
    capabilities: [
      {
        type: "storage",
        name: "Amazon S3",
        description: "Scalable object storage (Simple Storage Service)",
      },
      {
        type: "compute",
        name: "Amazon EC2",
        description: "Elastic Compute Cloud — CPU and GPU instances",
      },
    ],
    credential_fields: [
      {
        key: "access_key_id",
        label: "Access Key ID",
        required: true,
        secret: false,
        placeholder: "AKIA...",
      },
      {
        key: "secret_access_key",
        label: "Secret Access Key",
        required: true,
        secret: true,
        placeholder: "",
      },
    ],
    config_fields: [
      {
        key: "region",
        label: "Default Region",
        default: "eu-west-1",
        description: "AWS region (e.g. us-east-1, eu-west-1)",
      },
    ],
    last_checked_at: null,
    created_at: null,
    updated_at: null,
  },

  gcp: {
    id: "gcp",
    display_name: "Google Cloud Platform",
    description: "Global cloud platform — GCS, Compute Engine, and more",
    website: "https://cloud.google.com/",
    connector_type: "gcp",
    enabled: false,
    status: "error",
    status_message: "Authentication failed — invalid service account key",
    has_credentials: true,
    config: { zone: "europe-west1-b" },
    capabilities: [
      {
        type: "storage",
        name: "Cloud Storage",
        description: "Google Cloud Storage buckets",
      },
      {
        type: "compute",
        name: "Compute Engine",
        description: "CPU and GPU virtual machines",
      },
    ],
    credential_fields: [
      {
        key: "project_id",
        label: "Project ID",
        required: true,
        secret: false,
        placeholder: "my-project-123",
      },
      {
        key: "credentials_json",
        label: "Service Account JSON",
        required: true,
        secret: true,
        placeholder: "Paste the full JSON key file content",
      },
    ],
    config_fields: [
      {
        key: "zone",
        label: "Default Zone",
        default: "europe-west1-b",
        description: "Compute Engine zone (e.g. us-central1-a, europe-west1-b)",
      },
    ],
    last_checked_at: "2026-02-19T08:15:00Z",
    created_at: "2026-02-17T14:00:00Z",
    updated_at: "2026-02-19T08:15:00Z",
  },

  azure: {
    id: "azure",
    display_name: "Microsoft Azure",
    description:
      "Global cloud platform — Blob Storage, Virtual Machines, and more",
    website: "https://azure.microsoft.com/",
    connector_type: "azure",
    enabled: false,
    status: "unconfigured",
    status_message: "",
    has_credentials: false,
    config: {},
    capabilities: [
      {
        type: "storage",
        name: "Blob Storage",
        description: "Azure Blob Storage containers",
      },
      {
        type: "compute",
        name: "Virtual Machines",
        description: "CPU and GPU virtual machines",
      },
    ],
    credential_fields: [
      {
        key: "subscription_id",
        label: "Subscription ID",
        required: true,
        secret: false,
        placeholder: "",
      },
      {
        key: "tenant_id",
        label: "Tenant ID",
        required: true,
        secret: false,
        placeholder: "",
      },
      {
        key: "client_id",
        label: "Client ID (App ID)",
        required: true,
        secret: false,
        placeholder: "",
      },
      {
        key: "client_secret",
        label: "Client Secret",
        required: true,
        secret: true,
        placeholder: "",
      },
    ],
    config_fields: [
      {
        key: "region",
        label: "Default Region",
        default: "westeurope",
        description: "Azure region (e.g. westeurope, eastus, westus2)",
      },
      {
        key: "resource_group",
        label: "Resource Group",
        default: "",
        description:
          "Azure resource group name (optional, for scoped queries)",
      },
    ],
    last_checked_at: null,
    created_at: null,
    updated_at: null,
  },

  scaleway: {
    id: "scaleway",
    display_name: "Scaleway",
    description: "European cloud provider with GPU instances and S3 storage",
    website: "https://www.scaleway.com/en/",
    connector_type: "scaleway",
    enabled: false,
    status: "configured",
    status_message: "Credentials saved — run a connection test to activate",
    has_credentials: true,
    config: { zone: "fr-par-1" },
    capabilities: [
      {
        type: "storage",
        name: "Object Storage",
        description: "S3-compatible object storage (Scaleway Object Storage)",
      },
      {
        type: "compute",
        name: "Instances",
        description: "CPU and GPU compute instances (Scaleway Instances)",
      },
    ],
    credential_fields: [
      {
        key: "access_key",
        label: "Access Key",
        required: true,
        secret: false,
        placeholder: "SCW...",
      },
      {
        key: "secret_key",
        label: "Secret Key",
        required: true,
        secret: true,
        placeholder: "",
      },
      {
        key: "project_id",
        label: "Project ID",
        required: true,
        secret: false,
        placeholder: "",
      },
    ],
    config_fields: [
      {
        key: "zone",
        label: "Default Zone",
        default: "fr-par-1",
        description: "Availability zone (e.g. fr-par-1, nl-ams-1)",
      },
    ],
    last_checked_at: null,
    created_at: "2026-02-19T11:00:00Z",
    updated_at: "2026-02-19T11:00:00Z",
  },

  vastai: {
    id: "vastai",
    display_name: "Vast.ai",
    description: "GPU compute marketplace for deep learning",
    website: "https://vast.ai/",
    connector_type: "vastai",
    enabled: false,
    status: "unconfigured",
    status_message: "",
    has_credentials: false,
    config: {},
    capabilities: [
      {
        type: "compute",
        name: "GPU Instances",
        description: "On-demand and interruptible GPU instances",
      },
    ],
    credential_fields: [
      {
        key: "api_key",
        label: "API Key",
        required: true,
        secret: true,
        placeholder: "",
      },
    ],
    config_fields: [],
    last_checked_at: null,
    created_at: null,
    updated_at: null,
  },
};

/* ── Storage / Compute / Regions ─────────────────────────────────────────── */

export const MOCK_PROVIDER_STORAGE: Record<string, ProviderStorageOption[]> = {
  ovh: [
    {
      provider_id: "ovh",
      name: "ml-datasets",
      type: "object_storage",
      region: "GRA11",
      bytes_used: 5368709120,
      object_count: 234,
    },
    {
      provider_id: "ovh",
      name: "model-artifacts",
      type: "object_storage",
      region: "GRA11",
      bytes_used: 10737418240,
      object_count: 56,
    },
    {
      provider_id: "ovh",
      name: "training-checkpoints",
      type: "object_storage",
      region: "SBG5",
      bytes_used: 2147483648,
      object_count: 18,
    },
  ],
};

export const MOCK_PROVIDER_COMPUTE: Record<string, ProviderComputeInstance[]> = {
  ovh: [
    {
      provider_id: "ovh",
      name: "b2-30",
      vcpus: 8,
      memory_gb: 30,
      disk_gb: 200,
      gpu_type: null,
      gpu_count: 0,
      region: "GRA11",
      available: true,
    },
    {
      provider_id: "ovh",
      name: "b2-60",
      vcpus: 16,
      memory_gb: 60,
      disk_gb: 400,
      gpu_type: null,
      gpu_count: 0,
      region: "GRA11",
      available: true,
    },
    {
      provider_id: "ovh",
      name: "gpu-a100-80g",
      vcpus: 12,
      memory_gb: 120,
      disk_gb: 400,
      gpu_type: "A100",
      gpu_count: 1,
      region: "GRA11",
      available: true,
    },
    {
      provider_id: "ovh",
      name: "gpu-v100-32g",
      vcpus: 8,
      memory_gb: 64,
      disk_gb: 200,
      gpu_type: "V100",
      gpu_count: 1,
      region: "GRA11",
      available: true,
    },
  ],
};

export const MOCK_PROVIDER_REGIONS: Record<string, ProviderRegion[]> = {
  ovh: [
    { provider_id: "ovh", id: "GRA11", name: "Gravelines" },
    { provider_id: "ovh", id: "SBG5", name: "Strasbourg" },
    { provider_id: "ovh", id: "BHS5", name: "Beauharnois" },
    { provider_id: "ovh", id: "WAW1", name: "Warsaw" },
    { provider_id: "ovh", id: "UK1", name: "London" },
  ],
};

/* ── Connection test results ─────────────────────────────────────────────── */

export const MOCK_TEST_RESULTS: Record<string, ConnectionTestResult> = {
  ovh: {
    success: true,
    message: "Connected — 42 flavors available",
    latency_ms: 85.3,
  },
  infomaniak: {
    success: true,
    message: "Connected — 18 flavors available",
    latency_ms: 92.1,
  },
  aws: {
    success: true,
    message: "Connected — AWS account 123456789012",
    latency_ms: 120.1,
  },
  gcp: {
    success: false,
    message: "Authentication failed — invalid service account key",
    latency_ms: null,
  },
  azure: {
    success: true,
    message: "Connected — 156 VM sizes available in westeurope",
    latency_ms: 145.8,
  },
  scaleway: {
    success: true,
    message: "Connected to Scaleway API",
    latency_ms: 63.7,
  },
  vastai: {
    success: true,
    message: "Connected to Vast.ai API",
    latency_ms: 180.2,
  },
};
