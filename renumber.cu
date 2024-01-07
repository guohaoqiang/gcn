#include <assert.h>
#include <vector>
#include <string>
#include <ranges>
#include <algorithm>

extern "C"{

void dfs(int* dlrowPtr, int* dlcol, float* dlvals, int* vomp, int m, int n, int nnz){
  // Renumber the vertices based on a depth-first search of the graph in
  // dl, starting at vertex 0.

  printf("m = %d, n = %d, nnz = %d\n",m,n,nnz);
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

}
