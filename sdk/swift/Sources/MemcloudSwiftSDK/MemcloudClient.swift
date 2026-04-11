import Foundation

/// Memcloud Swift SDK — async/await native client for the Memcloud Memory API.
///
/// Usage:
/// ```swift
/// let client = MemcloudClient(apiKey: "mc_xxx")
/// try await client.store(text: "User prefers dark mode")
/// let results = try await client.search(query: "preferences")
/// let context = try await client.recall(query: "what do I know about the user")
/// ```
public actor MemcloudClient {

    // MARK: - Configuration

    public struct Config: Sendable {
        public let apiURL: String
        public let apiKey: String
        public let userId: String
        public let agentId: String
        public let timeout: TimeInterval

        public init(
            apiURL: String = "https://api.memcloud.dev/v1",
            apiKey: String,
            userId: String = "default",
            agentId: String = "swift-sdk",
            timeout: TimeInterval = 30
        ) {
            self.apiURL = apiURL.hasSuffix("/") ? String(apiURL.dropLast()) : apiURL
            self.apiKey = apiKey
            self.userId = userId
            self.agentId = agentId
            self.timeout = timeout
        }
    }

    // MARK: - Models

    public struct Memory: Codable, Sendable, Identifiable {
        public let id: String
        public let content: String
        public let memory_type: String?
        public let agent_id: String?
        public let pool_id: String?
        public let source_type: String?
        public let source_ref: String?
        public let structured_data: [String: AnyCodable]?
        public let confidence: Double?
        public let importance: Int?
        public let decay_score: Double?
        public let created_at: String?
        public let rrf_score: Double?
        public let rerank_score: Double?
        public let sources: [String]?
    }

    public struct StoreResponse: Codable, Sendable {
        public let status: String
        public let memories_created: Int?
        public let memory_ids: [String]?
        public let importance: Int?
    }

    public struct SearchResponse: Codable, Sendable {
        public let memories: [Memory]
    }

    public struct RecallResponse: Codable, Sendable {
        public let context: String
        public let format: String
        public let token_count: Int
        public let sections: [String: Int]
        public let latency_ms: Double
    }

    public struct BatchResponse: Codable, Sendable {
        public let processed: Int
        public let created: Int
        public let deleted: Int
        public let errors: [[String: AnyCodable]]?
    }

    public struct PoolResponse: Codable, Sendable {
        public let pool_id: String
        public let name: String?
        public let status: String?
        public let agents: [String]?
    }

    public struct PoolMemoriesResponse: Codable, Sendable {
        public let pool_id: String
        public let total: Int
        public let memories: [Memory]
    }

    public struct TemporalResponse: Codable, Sendable {
        public let total: Int
        public let memories: [Memory]
    }

    public struct GraphResponse: Codable, Sendable {
        public let start_entity: String
        public let nodes: Int
        public let edges: Int
        public let graph: GraphData
        public let linked_memories: [Memory]

        public struct GraphData: Codable, Sendable {
            public let nodes: [String]
            public let edges: [GraphEdge]
        }

        public struct GraphEdge: Codable, Sendable {
            public let source: String
            public let relation: String
            public let target: String
            public let memory_id: String?
            public let hop: Int?
        }
    }

    public struct HealthResponse: Codable, Sendable {
        public let status: String
        public let version: String
        public let postgres: Bool?
        public let redis: Bool?
        public let embedding_model: String?
    }

    // MARK: - Properties

    public let config: Config
    private let session: URLSession
    private var retryCount: Int = 3

    // MARK: - Init

    public init(config: Config) {
        self.config = config
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = config.timeout
        self.session = URLSession(configuration: sessionConfig)
    }

    public init(apiKey: String, apiURL: String = "https://api.memcloud.dev/v1", userId: String = "default", agentId: String = "swift-sdk") {
        self.init(config: Config(apiURL: apiURL, apiKey: apiKey, userId: userId, agentId: agentId))
    }

    // MARK: - Health

    public func health() async throws -> HealthResponse {
        try await get("/health")
    }

    // MARK: - Memory CRUD

    public func store(text: String, poolId: String? = nil, scope: String = "private", sourceType: String = "swift_sdk", metadata: [String: String]? = nil) async throws -> StoreResponse {
        var body: [String: Any] = [
            "text": text, "user_id": config.userId, "agent_id": config.agentId,
            "scope": scope, "source_type": sourceType
        ]
        if let poolId { body["pool_id"] = poolId }
        if let metadata { body["metadata"] = metadata }
        return try await post("/memories/", body: body)
    }

    public func search(query: String, topK: Int = 10, poolId: String? = nil) async throws -> SearchResponse {
        var body: [String: Any] = [
            "query": query, "user_id": config.userId, "agent_id": config.agentId, "top_k": topK
        ]
        if let poolId { body["pool_id"] = poolId }
        return try await post("/memories/search/", body: body)
    }

    public func recall(query: String? = nil, tokenBudget: Int = 4000, format: String = "markdown") async throws -> RecallResponse {
        var body: [String: Any] = [
            "user_id": config.userId, "agent_id": config.agentId,
            "token_budget": tokenBudget, "format": format,
            "include_profile": true, "include_recent": true
        ]
        if let query { body["query"] = query }
        return try await post("/recall", body: body)
    }

    public func deleteMemory(id: String) async throws {
        _ = try await request(method: "DELETE", path: "/memories/\(id)") as [String: AnyCodable]
    }

    // MARK: - Batch Operations (Phase 4.1)

    public func batchSync(operations: [[String: Any]]) async throws -> BatchResponse {
        let body: [String: Any] = [
            "user_id": config.userId, "agent_id": config.agentId,
            "operations": operations
        ]
        return try await post("/memories/batch", body: body)
    }

    // MARK: - Pool Management (Phase 4.2)

    public func createPool(poolId: String, name: String? = nil, agents: [String] = [], description: String? = nil) async throws -> PoolResponse {
        var body: [String: Any] = ["pool_id": poolId, "agents": agents]
        if let name { body["name"] = name }
        if let description { body["description"] = description }
        return try await post("/pools/", body: body)
    }

    public func getPoolMemories(poolId: String, limit: Int = 50, after: String? = nil, before: String? = nil) async throws -> PoolMemoriesResponse {
        var params = "limit=\(limit)"
        if let after { params += "&after=\(after)" }
        if let before { params += "&before=\(before)" }
        return try await get("/pools/\(poolId)/memories?\(params)")
    }

    public func writeToPool(poolId: String, text: String, sourceType: String = "pool_write") async throws -> StoreResponse {
        let body: [String: Any] = [
            "text": text, "user_id": config.userId, "agent_id": config.agentId,
            "source_type": sourceType
        ]
        return try await post("/pools/\(poolId)/memories", body: body)
    }

    public func deletePool(poolId: String) async throws {
        _ = try await request(method: "DELETE", path: "/pools/\(poolId)") as [String: AnyCodable]
    }

    // MARK: - Temporal Query (Phase 4.3)

    public func temporalQuery(relative: String? = nil, after: String? = nil, before: String? = nil, memoryType: String? = nil, limit: Int = 50) async throws -> TemporalResponse {
        var params = "user_id=\(config.userId)&limit=\(limit)"
        if let relative { params += "&relative=\(relative.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? relative)" }
        if let after { params += "&after=\(after)" }
        if let before { params += "&before=\(before)" }
        if let memoryType { params += "&memory_type=\(memoryType)" }
        return try await get("/memories/temporal?\(params)")
    }

    // MARK: - Graph Query (Phase 4.4)

    public func graphQuery(startEntity: String, traversal: String = "2-hop", relationshipTypes: [String]? = nil) async throws -> GraphResponse {
        var body: [String: Any] = [
            "start_entity": startEntity, "traversal": traversal, "user_id": config.userId
        ]
        if let types = relationshipTypes { body["relationship_types"] = types }
        return try await post("/graph/query", body: body)
    }

    // MARK: - HTTP Layer

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try await request(method: "GET", path: path)
    }

    private func post<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        try await request(method: "POST", path: path, body: body)
    }

    private func request<T: Decodable>(method: String, path: String, body: [String: Any]? = nil) async throws -> T {
        guard let url = URL(string: config.apiURL + path) else {
            throw MemcloudError.invalidURL(config.apiURL + path)
        }

        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let body {
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }

        // Retry with exponential backoff
        var lastError: Error?
        for attempt in 0..<retryCount {
            do {
                let (data, response) = try await session.data(for: req)
                guard let http = response as? HTTPURLResponse else {
                    throw MemcloudError.invalidResponse
                }
                guard (200...299).contains(http.statusCode) else {
                    let body = String(data: data, encoding: .utf8) ?? "Unknown"
                    throw MemcloudError.apiError(statusCode: http.statusCode, message: body)
                }
                return try JSONDecoder().decode(T.self, from: data)
            } catch let error as MemcloudError {
                throw error // Don't retry API errors
            } catch {
                lastError = error
                if attempt < retryCount - 1 {
                    let delay = UInt64(pow(2.0, Double(attempt))) * 1_000_000_000
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }
        throw lastError ?? MemcloudError.invalidResponse
    }
}

// MARK: - Errors

public enum MemcloudError: Error, LocalizedError {
    case invalidURL(String)
    case invalidResponse
    case apiError(statusCode: Int, message: String)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let url): return "Invalid Memcloud URL: \(url)"
        case .invalidResponse: return "Invalid response from Memcloud"
        case .apiError(let code, let msg): return "Memcloud API \(code): \(msg)"
        }
    }
}

// MARK: - Type-erased Codable helper

public struct AnyCodable: Codable, Sendable {
    public let value: Any

    public init(_ value: Any) { self.value = value }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Bool.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode([AnyCodable].self) { value = v.map(\.value) }
        else if let v = try? container.decode([String: AnyCodable].self) { value = v.mapValues(\.value) }
        else { value = NSNull() }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let v as Bool: try container.encode(v)
        case let v as Int: try container.encode(v)
        case let v as Double: try container.encode(v)
        case let v as String: try container.encode(v)
        default: try container.encodeNil()
        }
    }
}
