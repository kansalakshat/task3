// Deploys Verification.sol and writes deployment.json (address + ABI) so
// verify.py can reach the contract without parsing Hardhat's artifact tree.
const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

async function main() {
  const { name, chainId } = await hre.ethers.provider.getNetwork();
  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);

  console.log(`Network:  ${name} (chainId ${chainId})`);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Balance:  ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) {
    throw new Error("deployer has no ETH -- fund it from a Sepolia faucet, or use --network localhost");
  }

  const factory = await hre.ethers.getContractFactory("Verification");
  const contract = await factory.deploy();
  await contract.waitForDeployment();
  const address = await contract.getAddress();
  console.log(`Deployed Verification -> ${address}`);

  const artifact = await hre.artifacts.readArtifact("Verification");
  const out = {
    address,
    network: hre.network.name,
    chainId: Number(chainId),
    deployer: deployer.address,
    deployTx: contract.deploymentTransaction()?.hash ?? null,
    abi: artifact.abi,
  };
  const file = path.join(__dirname, "..", "deployment.json");
  fs.writeFileSync(file, JSON.stringify(out, null, 2));
  console.log(`Wrote ${file}`);
}

main().catch((e) => {
  console.error(e.message ?? e);
  process.exitCode = 1;
});
