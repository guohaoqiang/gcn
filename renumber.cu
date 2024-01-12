#include <assert.h>
#include <vector>
#include <map>
#include <string>
#include <ranges>
#include <algorithm>
#include "order_deg.cuh"
#include "order_rcm.cuh"
#include "order_gorder.cuh"
#include "edgelist.cuh"
#include "order_gorder.cuh"


template<typename T, typename T2>
inline bool set_max(T& accum, const T2& v)
{
  if ( v <= accum ) return false;
  accum = v;
  return true;
}
extern "C"{

void dfs(int* dlrowPtr, int* dlcol, float* dlvals, int* vomp, int m, int n, int nnz){
  // Renumber the vertices based on a depth-first search of the graph in
  // dl, starting at vertex 0.

  //printf("m = %d, n = %d, nnz = %d\n",m,n,nnz);
  std::string ertex_order_abbr("DFS");

  //assert( dl.rowPtr.size() == n + 1 );
  std::vector<unsigned int> dl_col;
  for (int i=0; i<nnz; ++i){
    dl_col.push_back(dlcol[i]);
  }
  auto dst_iter_make = [&](uint s)
  { return std::ranges::subrange(&dl_col[dlrowPtr[s]],&dl_col[dlrowPtr[s+1]]); };

  std::vector<uint> vo_to_dfs(n);  // original Vertex Order to DFS order
  vo_to_dfs[0] = n; // Will be changed back to zero.

  std::vector<unsigned int> rowPtr;
  std::vector<unsigned int> col;
  std::vector<float> vals;
  //col.resize( dl.col.size() );
  col.resize( nnz );
  //vals.resize( dl.col.size() );
  vals.resize( nnz );

  //
  // Perform Depth-First Search (DFS) on Each Component
  //

  rowPtr.reserve( n+1 );
  rowPtr.push_back( 0 );

  for ( int dfs_root_vo_idx = 0; dfs_root_vo_idx < n; )
    {
      auto root = dst_iter_make(dfs_root_vo_idx);
      std::vector< decltype(root) > stack { root };
      if ( dfs_root_vo_idx ) vo_to_dfs[ dfs_root_vo_idx ] = rowPtr.size() - 1;
      rowPtr.push_back( rowPtr.back() + root.size() );

      while ( !stack.empty() )
        {
          auto& dst_iter = stack.back();
          while ( dst_iter && vo_to_dfs[ dst_iter.front() ] )
            dst_iter.advance(1);
          if ( !dst_iter ) { stack.pop_back();  continue; }

          const uint dst_vo  = dst_iter.front();  dst_iter.advance(1);
          const uint dst_dfs = rowPtr.size() - 1;
          vo_to_dfs[ dst_vo ] = dst_dfs;
          auto dst_node_iterator = dst_iter_make( dst_vo );
          stack.push_back( dst_node_iterator );
          // Update edge list pointer. (Row Number to vals/col array index.)
          rowPtr.push_back( rowPtr.back() + dst_node_iterator.size() );
        }

      if ( rowPtr.size() > n ) break;

      // Find a vertex that has not been searched.
      while ( ++dfs_root_vo_idx < n && vo_to_dfs[dfs_root_vo_idx] );
      assert( dfs_root_vo_idx < n );
    }

  assert( rowPtr.size() == n + 1 );

  vo_to_dfs[0] = 0;

  std::vector<int> vo_mp;
  vo_mp.resize(m);
  for (int i=0; i<vo_to_dfs.size(); ++i){
      int v = vo_to_dfs[i];
      vo_mp[v] = i;
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto src_vo: std::views::iota(0,n) )
    {
      const auto src_dfs = vo_to_dfs[src_vo];
      const int d = dlrowPtr[src_vo+1] - dlrowPtr[src_vo];
      assert( rowPtr[src_dfs] + d == rowPtr[src_dfs+1] );

      // Sort destinations.  Tiling algorithm needs dests sorted.
      std::vector< std::pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dlrowPtr[ src_vo ];
      for ( auto e: std::views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dlvals[ e ], vo_to_dfs[ dlcol[ e ] ] );
      std::ranges::sort(perm, std::ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_dfs_i = rowPtr[src_dfs];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_dfs_i ] = dst_new;
          vals[ e_idx_dfs_i++ ] = val;
        }
    }
  //
  // Perform a rough test of whether the two graphs match.
  //

  std::vector<int64_t> check_vo(n);
  std::vector<int64_t> check_dfs(n);
  std::vector<double> checkw_vo(n);
  std::vector<double> checkw_dfs(n);

  for ( uint src_vo: std::views::iota(0,n) )
    {
      const auto src_dfs = vo_to_dfs[src_vo];
      const int d = dlrowPtr[src_vo+1] - dlrowPtr[src_vo];
      assert( rowPtr[src_dfs] + d == rowPtr[src_dfs+1] );
      const int inc = src_vo & 0xf;
      for ( auto n_idx: std::views::iota(0,d) )
        {
          const uint e_idx_vo = dlrowPtr[src_vo] + n_idx;
          const uint e_idx_dfs = rowPtr[src_dfs] + n_idx;
          check_vo[ dlcol[ e_idx_vo ] ] += inc;
          check_dfs[ col[ e_idx_dfs ] ] += inc;
          checkw_vo[ dlcol[ e_idx_vo ] ] += dlvals[ e_idx_vo ];
          checkw_dfs[ col[ e_idx_dfs ] ] += vals[ e_idx_dfs ];
        }
    }

  for ( uint src_vo: std::views::iota(0,n) )
    {
      assert( check_vo[src_vo] == check_dfs[ vo_to_dfs[src_vo] ] );
      assert( checkw_vo[src_vo] == checkw_dfs[ vo_to_dfs[src_vo] ] );
    }

  for (int i=0; i<rowPtr.size(); ++i)   dlrowPtr[i] = rowPtr[i];
  for (int i=0; i<col.size(); ++i)   dlcol[i] = col[i];
  for (int i=0; i<vals.size(); ++i)   dlvals[i] = vals[i];
  for (int i=0; i<vo_mp.size(); ++i)   vomp[i] = vo_mp[i];
}//end dfs

void gorder(int* dlrowPtr, int* dlcol, float* dlvals, int* vomp, int m, int n, int nnz)
{
  std::string vertex_order_abbr("GOR");

  //assert( dl.rowPtr.size() == n + 1 );

  std::vector<unsigned int> vo_to_gorder;  // original Vertex Order to Gorder order

  std::vector<unsigned int> rowPtr;
  std::vector<unsigned int> col;
  std::vector<float> vals;
  //col.resize( dl.col.size() );
  col.resize( nnz );
  //vals.resize( dl.col.size() );
  vals.resize( nnz );

  //
  // Convert CSR to edge lists
  //
  ul window_sz= 3;
  Edgelist h(dlrowPtr, dlcol, m, nnz);
  vo_to_gorder.reserve(m);
  vo_to_gorder = complete_gorder(h, window_sz);

  //
  // According to renumbered vertex order, generate rowPtr 
  //  
  std::vector<int> vo_mp;
  vo_mp.resize(m);
  for (ul i=0; i<vo_to_gorder.size(); ++i){
      ul v = vo_to_gorder[i];
      vo_mp[v] = i;
  }
  rowPtr.push_back( 0 );
  for (auto v:vo_mp){
    rowPtr.push_back(rowPtr.back()+dlrowPtr[v+1]-dlrowPtr[v]);
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  
  for ( auto src_vo: std::views::iota(0,n) )
    {
      const auto src_gorder = vo_to_gorder[src_vo];
      const int d = dlrowPtr[src_vo+1] - dlrowPtr[src_vo];
      assert( rowPtr[src_gorder] + d == rowPtr[src_gorder+1] );

      // Sort destinations.  Tiling algorithm needs dests sorted.
      std::vector< std::pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dlrowPtr[ src_vo ];
      for ( auto e: std::views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dlvals[ e ], vo_to_gorder[ dlcol[ e ] ] );
      std::ranges::sort(perm, std::ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_gorder_i = rowPtr[src_gorder];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_gorder_i ] = dst_new;
          vals[ e_idx_gorder_i++ ] = val;
        }
    }

    if ( false ){
        //print_ord(vertex_order_abbr, vo_to_gorder, rowPtr, col);
    }
  //
  // Perform a rough test of whether the two graphs match.
  //

  for (int i=0; i<rowPtr.size(); ++i)   dlrowPtr[i] = rowPtr[i];
  for (int i=0; i<col.size(); ++i)   dlcol[i] = col[i];
  for (int i=0; i<vals.size(); ++i)   dlvals[i] = vals[i];
  for (int i=0; i<vo_mp.size(); ++i)   vomp[i] = vo_mp[i];
}// end GOR


void perm_apply(int* dlrowPtr, int* dlcol, float* dlvals, int* vomp, int m, int n, int nnz)
{

  std::vector<unsigned int> rowPtr;
  std::vector<unsigned int> col;
  std::vector<float> vals;
  //col.resize( dl.col.size() );
  col.resize( nnz );
  //vals.resize( dl.col.size() );
  vals.resize( nnz );
  
  rowPtr.reserve( n+1 );
  rowPtr.push_back( 0 );
  std::vector<unsigned int> vold_to_new(n,n);

  for ( auto v_new: std::views::iota(0,n) )
    {
      const int v_old = vomp[ v_new ];
      assert( vold_to_new[ v_old ] == n );
      vold_to_new[ v_old ] = v_new;
      rowPtr.push_back( rowPtr.back() + dlrowPtr[v_old+1] - dlrowPtr[v_old] );
    }


  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto v_old: std::views::iota(0,n) )
    {
      const auto v_new = vold_to_new[v_old];
      const int d = dlrowPtr[v_old+1] - dlrowPtr[v_old];

      // Sort destinations.  Tiling algorithm needs dests sorted.
      std::vector< std::pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_old = dlrowPtr[ v_old ];
      for ( auto e: std::views::iota( e_idx_old, e_idx_old + d ) )
        perm.emplace_back( dlvals[ e ], vold_to_new[ dlcol[ e ] ] );
      std::ranges::sort(perm, std::ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_new_i = rowPtr[v_new];
      for ( auto [val, dst_new]: perm )
        {
          col[ e_idx_new_i ] = dst_new;
          vals[ e_idx_new_i++ ] = val;
        }
    }

    if ( false ){
        //print_ord(this->vertex_order_abbr, vold_to_new, this->rowPtr, col);
    }
  //
  // Perform a rough test of whether the two graphs match.
  //

    std::vector<int64_t> check_old(n);
    std::vector<int64_t> check_new(n);
    std::vector<double> checkw_old(n);
    std::vector<double> checkw_new(n);

  for ( uint v_old: std::views::iota(0,n) )
    {
      const auto v_new = vold_to_new[v_old];
      const int d = dlrowPtr[v_old+1] - dlrowPtr[v_old];
      assert( rowPtr[v_new] + d == rowPtr[v_new+1] );
      const int inc = v_old & 0xf;
      for ( auto n_idx: std::views::iota(0,d) )
        {
          const uint e_idx_old = dlrowPtr[v_old] + n_idx;
          const uint e_idx_new = rowPtr[v_new] + n_idx;
          check_old[ dlcol[ e_idx_old ] ] += inc;
          check_new[ col[ e_idx_new ] ] += inc;
          checkw_old[ dlcol[ e_idx_old ] ] += dlvals[ e_idx_old ];
          checkw_new[ col[ e_idx_new ] ] += vals[ e_idx_new ];
        }
    }

  for ( uint v_old: std::views::iota(0,n) )
    {
      assert( check_old[v_old] == check_new[ vold_to_new[v_old] ] );
      assert( checkw_old[v_old] == checkw_new[ vold_to_new[v_old] ] );
    }
  
  for (int i=0; i<rowPtr.size(); ++i)   dlrowPtr[i] = rowPtr[i];
  for (int i=0; i<col.size(); ++i)   dlcol[i] = col[i];
  for (int i=0; i<vals.size(); ++i)   dlvals[i] = vals[i];
}
void rabbit(int* dlrowPtr, int* dlcol, float* dlvals, int* vomp, int m, int n, int nnz)
{
  /// Order vertices based on modularity, implementing several variations.
  //
  // Original Iterative Serial Algorithm: 
  //   Shiokawa 13 AAAI "Fast algorithm for modularity based clustering."
  //   https://aaai.org/papers/455-fast-algorithm-for-modularity-based- graph-clustering/
  //   Set opt_iterative = true;
  //
  // Parallel Implementation. Rabbit properly refers to the parallel version.
  //   Arai 16 IPDPS
  //   https://ieeexplore.ieee.org/document/7515998
  //
  // Idea for Hub Grouping and Sorting
  //   Balaji 23 ISPASS
  //   https://ieeexplore.ieee.org/document/10158154
  //   Set opt_h

  std::string vertex_order_abbr("RBT");

  // Variations from Balaji 23 ISPASS 
  //
  const bool opt_hub_group = false;
  const bool opt_hub_sort = false;

  // When true, operate in degree order of current set of vertices.
  // When true, closer to Shiokawa 17 AAAI.
  const bool opt_iterative = true;

  //assert( dl.rowPtr.size() == n + 1 );
  std::vector<unsigned int> dl_col;
  for (int i=0; i<nnz; ++i){
    dl_col.push_back(dlcol[i]);
  }
  auto dst_iter_make = [&](uint s)
  { return std::ranges::subrange(&dl_col[dlrowPtr[s]],&dl_col[dlrowPtr[s+1]]); };

  struct Tree_Node {
    Tree_Node(Tree_Node *a, Tree_Node *b):lchild(a),rchild(b),v_idx(-1){}
    Tree_Node(int v):lchild(nullptr),rchild(nullptr),v_idx(v){}
    Tree_Node(){}
    Tree_Node *lchild, *rchild;
    int v_idx;
    void leaves_apply( std::vector<int>& perm ) {
      if ( lchild ) { lchild->leaves_apply(perm); rchild->leaves_apply(perm); }
      else          { perm.push_back( v_idx ); } }
  };

  struct Vertex {
      std::map<int,int> dst_wht; // NOT the graphs original edge weight.
    Tree_Node leaf_node, cluster_node, *tree_node;
    int deg, deg_orig, round;
  };

  std::vector<uint> v_this_round(n);
  std::vector<Vertex> mgraph(n);
  int n_edges = 0;

  // If true, perform clustering on a directed version of the graph.
  const bool force_undirected = true;

  // Prepare structure used for Rabbit's community detection.
  //
  for ( auto v: std::views::iota(0,n) )
    {
      Vertex& vo = mgraph[v];
      // This edge weight is used only for computing modularity. 
      for ( auto d: dst_iter_make(v) ) 
        if ( d != v )
          {
            vo.dst_wht[d] = 1;
            if ( force_undirected ) mgraph[d].dst_wht[v] = 1;
          }
      vo.deg_orig = vo.deg = vo.dst_wht.size();
      n_edges += vo.deg;
      vo.leaf_node = Tree_Node(v);
      vo.tree_node = &vo.leaf_node;
      v_this_round[v] = v;
      vo.round = 0;
    }

  // Note: cluster_shyness = 1 is the value used in Arai 16.
  const double opt_cluster_shyness = 1;
  const double two_m_inv = opt_cluster_shyness / double( 2 * n_edges );

  std::vector<uint> v_next_round;

  for ( int round = 1; !v_this_round.empty(); round++ )
    {
        std::ranges::sort
        ( v_this_round, std::ranges::less(), [&](auto i){ return mgraph[i].deg; });

      if ( opt_iterative )
        printf("Rabbit round %2d, n elts %zd\n",round,v_this_round.size());

      for ( auto u: v_this_round )
        {
          Vertex& uo = mgraph[u];
          if ( opt_iterative && uo.round == round ) continue;

          // Find neighbor of u with the largest change in modularity. (Delta Q)
          double dQ_max = -1;
          int v = -1;
          const double dv_2m = uo.deg * two_m_inv;
          for ( auto [d,w]: uo.dst_wht )
            if ( set_max( dQ_max, w - mgraph[d].deg * dv_2m ) ) v = d;
          if ( dQ_max <= 0 ) continue;

          // Modularity improves, so u is merged into v.
          //
          Vertex& vo = mgraph[v];
          vo.deg += uo.deg;

          // Update links affected by "removal" of u.
          for ( auto [d,w]: uo.dst_wht )
            {
              if ( d == v ) continue;
              vo.dst_wht[d] += w;
              auto& dodw = mgraph[d].dst_wht;
              if ( !dodw.contains(u) ) continue;
              dodw[v] += dodw[u];
              dodw.erase(u);
            }
          vo.dst_wht.erase(u);

          // Add to dendrogram for this cluster.
          uo.cluster_node = Tree_Node(vo.tree_node,uo.tree_node);
          uo.tree_node = nullptr;
          vo.tree_node = &uo.cluster_node;

          if ( !opt_iterative || vo.round == round ) continue;
          vo.round = round;
          v_next_round.push_back(v);
        }

      if ( !opt_iterative ) break;
      assert( v_next_round.size() < v_this_round.size() );
      swap(v_this_round,v_next_round);
      v_next_round.clear();
    }

  // Sanity Check
  int deg_sum = 0;
  for ( auto& vo: mgraph ) if ( vo.tree_node ) deg_sum += vo.deg;
  assert( deg_sum == n_edges );

  int n_communities = 0;
  int n_hub_edges = 0;
  int n_hub_vertices = 0;
  std::vector<int> vo_to_community(n);
  std::vector<int> perm_rbt; perm_rbt.reserve(n);

  // Compute Modularity. This time keep the 1/(2m) factor. (m is n_edges)
  double q = 0;
  const double twom_inv = 1.0 / ( 2 * n_edges );
  const double twom_inv_sq = twom_inv * twom_inv;

  // Traverse dendrograms.
  for ( auto& vo: mgraph )
    if ( vo.tree_node )
      {
        int w_total = 0;
        for ( auto [_,w]: vo.dst_wht ) w_total += w;
        q += ( vo.deg - w_total ) * twom_inv - vo.deg * vo.deg * twom_inv_sq;
        const int c_idx = ++n_communities;
        const auto c_start = perm_rbt.size();
        vo.tree_node->leaves_apply(perm_rbt); // Append perm_rbt with leaves.
        const auto c_end = perm_rbt.size();
        for ( auto v_new: std::views::iota(c_start,c_end) )
          vo_to_community[ perm_rbt[v_new] ] = c_idx;
      }

  // Optionally apply special placement for hub (inter-community) nodes.
  //
  std::vector<int> vo_mp;
  vo_mp.reserve(n);
  std::vector<int> v_old_hub; v_old_hub.reserve(n/2);
  for ( auto v_new: std::views::iota(0,n) )
    {
      const auto v_old = perm_rbt[v_new];
      const auto c_idx = vo_to_community[ v_old ];
      int n_hub_edges_here = 0;
      for ( auto d: dst_iter_make(v_old) )
        if ( vo_to_community[d] != c_idx ) n_hub_edges_here++;
      n_hub_edges += n_hub_edges_here;
      if ( n_hub_edges_here ) n_hub_vertices++;
      if ( opt_hub_group && n_hub_edges_here ) v_old_hub.push_back( v_old );
      else                                     vo_mp.push_back( v_old );
    }
  if ( opt_hub_sort && opt_hub_group )
    std::ranges::sort
      ( v_old_hub, std::ranges::less(), [&](auto i){ return mgraph[i].deg_orig; });

  for ( auto v: v_old_hub ) vo_mp.push_back( v );

  printf("Shyness %.1f. Iter %d  GH %d GS %d "
         "Rabbit found %d communities, %d hubs %.3f%%, edges %d. Mod %f\n",
         opt_cluster_shyness, opt_iterative, opt_hub_group, opt_hub_sort,
         n_communities, n_hub_vertices,
         100.0 * n_hub_vertices / n, n_hub_edges,q);

  for (int i=0; i<vo_mp.size(); ++i)   vomp[i] = vo_mp[i];
  perm_apply(dlrowPtr, dlcol, dlvals, vomp, m, n, nnz);
} // end rabbit



}
