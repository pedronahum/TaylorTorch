import Foundation
import Torch
import _Differentiation

/// Deterministic xoshiro-style generator used to make RL rollouts reproducible.
struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64

  /// Creates a generator seeded with the provided value.
  init(seed: UInt64) {
    state = seed
  }

  /// Produces the next pseudo-random `UInt64`.
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}

var rng = SeededGenerator(seed: 0xdead_beef)

typealias Observation = Tensor
typealias Reward = Float

/// Minimal reinforcement-learning environment contract.
protocol Environment {
  associatedtype Action: Equatable
  /// Applies `action` and returns the resulting observation and reward.
  mutating func step(with action: Action) -> (observation: Observation, reward: Reward)
  /// Resets the environment to its initial state and returns the starting observation.
  mutating func reset() -> Observation
}

/// Protocol adopted by agents that select the next action.
protocol Agent: AnyObject {
  associatedtype Action: Equatable
  /// Chooses the next action given the current observation and previous reward.
  func step(observation: Observation, reward: Reward) -> Action
}

/// Discrete actions available in the catch game.
enum CatchAction: Int {
  case none
  case left
  case right
}

/// Convenience struct representing grid coordinates.
struct Position: Equatable, Hashable {
  var x: Int
  var y: Int
}

/// Grid world where a paddle attempts to catch a falling ball.
struct CatchEnvironment: Environment {
  typealias Action = CatchAction

  let rowCount: Int
  let columnCount: Int
  var ballPosition: Position
  var paddlePosition: Position

  init(rowCount: Int, columnCount: Int) {
    self.rowCount = rowCount
    self.columnCount = columnCount
    self.ballPosition = Position(x: 0, y: 0)
    self.paddlePosition = Position(x: 0, y: 0)
    _ = reset()
  }

  /// Applies the supplied action, advances the ball, and returns the new observation plus reward.
  mutating func step(with action: CatchAction) -> (observation: Observation, reward: Reward) {
    switch action {
    case .left where paddlePosition.x > 0:
      paddlePosition.x -= 1
    case .right where paddlePosition.x < columnCount - 1:
      paddlePosition.x += 1
    default:
      break
    }

    ballPosition.y += 1
    let currentReward = reward

    if ballPosition.y == rowCount {
      return (reset(), currentReward)
    }

    return (observation, currentReward)
  }

  @discardableResult
  /// Reinitialises the ball and paddle positions.
  mutating func reset() -> Observation {
    let randomColumn = Int.random(in: 0..<columnCount, using: &rng)
    ballPosition = Position(x: randomColumn, y: 0)
    paddlePosition = Position(x: columnCount / 2, y: rowCount - 1)
    return observation
  }

  /// Scalar reward: +1 for successful catch, -1 for a miss, 0 otherwise.
  var reward: Reward {
    guard ballPosition.y == rowCount else {
      return 0
    }
    return abs(ballPosition.x - paddlePosition.x) <= 1 ? 1 : -1
  }

  /// Observation vector normalised to `[0, 1]`.
  var observation: Observation {
    Tensor(
      array: [
        Float(ballPosition.x) / Float(columnCount),
        Float(ballPosition.y) / Float(rowCount),
        Float(paddlePosition.x) / Float(columnCount),
      ],
      shape: [3]
    )
  }

  /// Binary occupancy grid useful for printing the board.
  var grid: Observation {
    var scalars = [Float](repeating: 0, count: rowCount * columnCount)
    if ballPosition.y >= 0, ballPosition.y < rowCount,
      ballPosition.x >= 0, ballPosition.x < columnCount
    {
      scalars[ballPosition.y * columnCount + ballPosition.x] = 1
    }
    if paddlePosition.y >= 0, paddlePosition.y < rowCount,
      paddlePosition.x >= 0, paddlePosition.x < columnCount
    {
      scalars[paddlePosition.y * columnCount + paddlePosition.x] = 1
    }
    return Tensor(array: scalars, shape: [rowCount, columnCount])
  }
}

extension CatchEnvironment: CustomStringConvertible {
  /// Textual representation of the grid with ball and paddle positions.
  var description: String {
    let scalars = grid.toArray(as: Float.self)
    var rows: [String] = []
    rows.reserveCapacity(rowCount)
    for row in 0..<rowCount {
      let start = row * columnCount
      let slice = scalars[start..<(start + columnCount)]
      rows.append(slice.map { $0 > 0 ? "1" : "0" }.joined(separator: " "))
    }
    return rows.joined(separator: "\n")
  }
}

/// Policy-gradient agent (REINFORCE with a moving baseline) trained online.
final class CatchAgent: Agent {
  typealias Action = CatchAction
  typealias Policy = SequentialBlock<Sequential<Dense, Dense>>

  var model: Policy
  var optimizer: AdamW<Policy>
  var previousObservation: Observation?
  var previousActionIndex: Int?
  var baseline: Float
  let baselineMomentum: Float

  /// Builds a two-layer MLP policy and initialises the running reward baseline.
  /// - Parameters:
  ///   - initialReward: Reward at the moment the agent is created (used for baseline warm start).
  ///   - learningRate: AdamW learning rate.
  init(initialReward: Reward, learningRate: Double) {
    self.model = SequentialBlock {
      Dense(inFeatures: 3, outFeatures: 50, activation: .relu)
      Dense(inFeatures: 50, outFeatures: 3, activation: .identity)
    }
    self.optimizer = AdamW(for: model, learningRate: learningRate)
    self.previousObservation = nil
    self.previousActionIndex = nil
    self.baseline = initialReward
    self.baselineMomentum = 0.05
  }

  /// Performs a REINFORCE update using the previous transition and chooses the next action.
  /// - Parameters:
  ///   - observation: Current observation from the environment.
  ///   - reward: Reward obtained after executing the previous action.
  /// - Returns: The action that should be taken next.
  func step(observation: Observation, reward: Reward) -> Action {
    if let storedObservation = previousObservation, let storedActionIndex = previousActionIndex {
      let advantage = reward - baseline
      let (_, pullback) = valueWithPullback(at: model) { current -> Tensor in
        let logits = current(storedObservation.unsqueezed(dim: 0))
        let logProbs = self.logSoftmax(logits)
        let selectedLogProb = logProbs.indexSelect(dim: 1, indices: [Int64(storedActionIndex)])
          .sum()
        return selectedLogProb.negated().multiplying(Tensor(advantage, dtype: .float32))
      }
      let grad = pullback(Tensor(1.0, dtype: .float32))
      optimizer.update(&model, along: grad)
    }

    baseline += (reward - baseline) * baselineMomentum

    let input = observation.unsqueezed(dim: 0)
    let logits = model(input)
    let logProbs = logSoftmax(logits)
    let probabilities = exp(logProbs)
    let actionIndex = sampleAction(from: probabilities)

    previousObservation = observation
    previousActionIndex = actionIndex

    return Action(rawValue: actionIndex) ?? .none
  }

  /// Computes the optimal action assuming perfect knowledge of the environment dynamics.
  func perfectAction(for observation: Observation) -> Action {
    let scalars = observation.toArray(as: Float.self)
    guard scalars.count == 3 else { return .none }
    let ballX = scalars[0]
    let paddleX = scalars[2]
    if abs(ballX - paddleX) < Float.ulpOfOne { return .none }
    return paddleX < ballX ? .right : .left
  }

  /// Returns a uniformly random action.
  func randomAction() -> Action {
    let id = Int.random(in: 0..<3, using: &rng)
    return Action(rawValue: id) ?? .none
  }

  /// Numerically stable log-softmax helper.
  private func logSoftmax(_ logits: Tensor) -> Tensor {
    let maxValues = logits.max(dim: 1, keepdim: true).values
    let shifted = logits - maxValues
    let logSumExp = log(exp(shifted).sum(dim: 1, keepdim: true))
    return shifted - logSumExp
  }

  /// Draws an action index from the categorical distribution represented by `probabilities`.
  private func sampleAction(from probabilities: Tensor) -> Int {
    let scalars = probabilities[0].toArray(as: Float.self)
    let randomValue = Float.random(in: 0..<1, using: &rng)
    var cumulative: Float = 0
    for (index, probability) in scalars.enumerated() {
      cumulative += probability
      if randomValue <= cumulative {
        return index
      }
    }
    return scalars.indices.last ?? 0
  }
}

/// Five-by-five board with a single falling ball and paddle.
var environment = CatchEnvironment(rowCount: 5, columnCount: 5)
/// Online policy-gradient agent trained while interacting with the environment.
var agent = CatchAgent(initialReward: environment.reward, learningRate: 0.05)
/// Latest action executed in the environment.
var action: CatchAction = .none

var gameCount = 0
var winCount = 0
var totalWinCount = 0
let maxIterations = 5_000

while gameCount < maxIterations {
  /// Advance the simulation using the previous action and let the agent react to the new observation.
  let (observation, reward) = environment.step(with: action)
  action = agent.step(observation: observation, reward: reward)

  if reward != 0 {
    gameCount += 1
    if reward > 0 {
      winCount += 1
      totalWinCount += 1
    }
    if gameCount % 20 == 0 {
      let recentRate = Float(winCount) / 20.0
      let totalRate = Float(totalWinCount) / Float(gameCount)
      print("Win rate (last 20 games): \(recentRate)")
      print("Win rate (total): \(totalRate) [\(totalWinCount)/\(gameCount)]")
      winCount = 0
    }
  }
}

let finalRate = Float(totalWinCount) / Float(gameCount)
/// Summarise the agent's success rate over the full run.
print("Win rate (final): \(finalRate) [\(totalWinCount)/\(gameCount)]")
