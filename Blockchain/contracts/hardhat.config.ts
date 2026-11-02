import { defineConfig } from "hardhat/config";
import toolboxMochaEthers from "@nomicfoundation/hardhat-toolbox-mocha-ethers";
import hardhatIgnition from "@nomicfoundation/hardhat-ignition";
import dotenv from "dotenv";

dotenv.config();

const PRIVATE_KEY =
  process.env.PRIVATE_KEY ||
  "0x0000000000000000000000000000000000000000000000000000000000000000";

export default defineConfig({
  plugins: [toolboxMochaEthers, hardhatIgnition],

  solidity: {
    version: "0.8.20",
  },

  networks: {
    monadTestnet: {
      type: "http",
      url: "https://testnet-rpc.monad.xyz",
      chainId: 10143,
      accounts: [PRIVATE_KEY],
    },
  },
});