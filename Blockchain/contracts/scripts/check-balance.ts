import { network } from "hardhat";

const { ethers } = await network.create({
  network: "monadTestnet",
});

const [signer] = await ethers.getSigners();

console.log("Wallet:", signer.address);

const balance = await ethers.provider.getBalance(signer.address);

console.log("Balance:", ethers.formatEther(balance), "MON");