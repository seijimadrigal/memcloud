// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MemcloudSwiftSDK",
    platforms: [.macOS(.v15), .iOS(.v17)],
    products: [
        .library(name: "MemcloudSwiftSDK", targets: ["MemcloudSwiftSDK"]),
    ],
    targets: [
        .target(name: "MemcloudSwiftSDK"),
        .testTarget(name: "MemcloudSwiftSDKTests", dependencies: ["MemcloudSwiftSDK"]),
    ]
)
