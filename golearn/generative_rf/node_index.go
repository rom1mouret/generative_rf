package generative_rf

import (
  "github.com/sjwhitworth/golearn/trees"
)


func depth(node *trees.DecisionTreeNode) int {
  if node.Children == nil {
    return 1
  }

  maxDepth := 0
  for _, child := range node.Children {
    depth := depth(child)
    if depth > maxDepth {
      maxDepth = depth
    }
  }

  return maxDepth + 1
}

func maxNodeIndex(tree *trees.ID3DecisionTree) int {
  return (1 << (depth(tree.Root) + 1)) - 2  // i.e 2^(depth+1) - 2
}

type NodeIndex struct {
  Current int
}

func (nodeIndex NodeIndex) Left() int {
  return nodeIndex.Current*2 + 1
}

func (nodeIndex *NodeIndex) Right() int {
  return nodeIndex.Current*2 + 2
}

func (nodeIndex *NodeIndex) MoveLeft() int {
  nodeIndex.Current = nodeIndex.Left()
  return nodeIndex.Current
}

func (nodeIndex *NodeIndex) MoveRight() int {
  nodeIndex.Current = nodeIndex.Right()
  return nodeIndex.Current
}
