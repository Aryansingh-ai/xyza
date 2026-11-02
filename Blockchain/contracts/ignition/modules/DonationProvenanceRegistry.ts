import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const DonationProvenanceRegistryModule = buildModule(
  "DonationProvenanceRegistryModule",
  (m) => {
    const registry = m.contract("DonationProvenanceRegistry");

    return { registry };
  }
);

export default DonationProvenanceRegistryModule;