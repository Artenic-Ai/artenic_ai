import type {
  ProviderComputeInstance,
  ProviderDetail,
  ProviderRegion,
  ProviderStorageOption,
  ProviderSummary,
} from "@/types/api";

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
    id: "aws",
    display_name: "Amazon Web Services",
    description: "Global cloud platform by Amazon",
    enabled: false,
    status: "unconfigured",
    capabilities: [
      {
        type: "storage",
        name: "S3",
        description: "Object storage",
      },
      {
        type: "compute",
        name: "EC2",
        description: "Elastic Compute Cloud",
      },
    ],
  },
  {
    id: "gcp",
    display_name: "Google Cloud Platform",
    description: "Cloud services by Google",
    enabled: false,
    status: "error",
    capabilities: [
      {
        type: "storage",
        name: "Cloud Storage",
        description: "Object storage",
      },
      {
        type: "compute",
        name: "Compute Engine",
        description: "Virtual machines",
      },
    ],
  },
];

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
};

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
