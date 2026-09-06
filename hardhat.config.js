require("@nomicfoundation/hardhat-ethers");
require("dotenv").config();

const { SEPOLIA_RPC_URL, PRIVATE_KEY } = process.env;

// The sepolia network is only registered when both an RPC URL and a key are
// present, so `--network sepolia` fails loudly rather than silently deploying
// somewhere unexpected. With them unset, use `--network localhost`.
const networks = {};
if (SEPOLIA_RPC_URL && PRIVATE_KEY) {
  networks.sepolia = {
    url: SEPOLIA_RPC_URL,
    accounts: [PRIVATE_KEY.startsWith("0x") ? PRIVATE_KEY : `0x${PRIVATE_KEY}`],
    chainId: 11155111,
  };
}

module.exports = {
  solidity: "0.8.24",
  networks,
};
