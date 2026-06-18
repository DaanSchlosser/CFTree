#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/IO/PLY.h>  // write_PLY

#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

using K     = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_3;
using Mesh  = CGAL::Surface_mesh<K::Point_3>;

static bool read_xyz(const std::string& path, std::vector<Point>& pts) {
  std::ifstream fin(path);
  if (!fin) return false;
  double x, y, z;
  // Read triples (skips comments/blank lines automatically if formatted)
  while (fin >> x >> y >> z) {
    pts.emplace_back(x, y, z);
  }
  return !pts.empty();
}

// Wrap one point cloud and write the PLY to a file. This is the single source
// of truth for the wrapping computation and the file output: BOTH the
// single-shot CLI path and the persistent --server path call it, so their
// output is byte-identical by construction. Returns false (with `err` set) on
// any failure. The output stream is flushed and closed before returning, so a
// caller that observes success can immediately read a complete file.
static bool wrap_one(const std::string& in, double ralpha, double roffset,
                     const std::string& out, std::string& err) {
  std::vector<Point> pts;
  if (!read_xyz(in, pts)) { err = "read failed: " + in; return false; }

  auto bb = CGAL::bbox_3(pts.begin(), pts.end());
  const double dx = bb.xmax() - bb.xmin();
  const double dy = bb.ymax() - bb.ymin();
  const double dz = bb.zmax() - bb.zmin();
  const double diag   = std::sqrt(dx*dx + dy*dy + dz*dz);
  const double alpha  = diag / ralpha;
  const double offset = alpha / roffset;

  Mesh wrap;
  CGAL::alpha_wrap_3(pts, alpha, offset, wrap);

  std::ofstream fout(out, std::ios::binary);
  if (!fout) { err = "cannot open output file: " + out; return false; }
  CGAL::IO::write_PLY(fout, wrap);
  fout.flush();
  fout.close();
  if (!fout) { err = "write failed: " + out; return false; }
  return true;
}

// Split a line on tab characters into exactly `n` fields. Returns false if the
// field count differs. A trailing '\r' (Windows line ending) is tolerated.
static bool split_tab(std::string line, std::vector<std::string>& out, size_t n) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
  out.clear();
  size_t start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == '\t') {
      out.push_back(line.substr(start, i - start));
      start = i + 1;
    }
  }
  return out.size() == n;
}

// Persistent "server" mode: amortizes process creation + dynamic-link + CGAL
// static init (~10ms native) across many trees instead of paying it per tree.
//
// Protocol (strict request/response lockstep, one command -> one status line):
//   stdin : "<in_xyz>\t<ralpha>\t<roffset>\t<out_ply>\n" per tree
//   stdout: "OK\n" on success, or "ERR\t<reason>\n" on a per-tree failure
//   stderr: not used (the caller redirects it to /dev/null)
//   EOF on stdin -> clean exit (no QUIT command).
//
// stdout carries ONLY status lines so the caller stays in sync. The PLY is
// written to the file named in the command (flushed+closed before "OK"), never
// streamed over stdout. The wrap itself goes through the same wrap_one() as the
// single-shot path, so output is byte-identical regardless of how many trees a
// single process has already handled.
static int run_server() {
  std::ios::sync_with_stdio(false);  // speeds the command channel; cannot affect file output
  std::string line;
  std::vector<std::string> f;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (!split_tab(line, f, 4)) {
      std::cout << "ERR\tmalformed command\n" << std::flush;
      continue;
    }
    double ralpha, roffset;
    try {
      ralpha  = std::stod(f[1]);
      roffset = std::stod(f[2]);
    } catch (...) {
      std::cout << "ERR\tbad params\n" << std::flush;
      continue;
    }
    std::string err;
    if (wrap_one(f[0], ralpha, roffset, f[3], err)) {
      std::cout << "OK\n" << std::flush;
    } else {
      std::cout << "ERR\t" << err << "\n" << std::flush;
    }
  }
  return 0;
}

int main(int argc, char** argv){
  // Persistent coprocess mode. Everything below this point is the original
  // single-shot CLI, unchanged, so its output stays byte-identical.
  if (argc >= 2 && std::string(argv[1]) == "--server") {
    return run_server();
  }

  if(argc < 2){
    std::cerr << "Usage: awrap_points <input.xyz> [ralpha=20] [roffset=50] [out.ply|-]\n";
    std::cerr << "       awrap_points --server   (persistent: reads '<in>\\t<ralpha>\\t<roffset>\\t<out>' lines on stdin)\n";
    return 2;
  }
  const std::string in  = argv[1];
  const double ralpha   = (argc > 2) ? std::stod(argv[2]) : 15.0;
  const double roffset  = (argc > 3) ? std::stod(argv[3]) : 50.0;
  const std::string out = (argc > 4) ? argv[4] : "-";

  if(out == "-"){
    // Stream PLY to stdout (manual/debug use; the pipeline always passes a path).
    std::vector<Point> pts;
    if(!read_xyz(in, pts)){
      std::cerr << "Failed to read XYZ points from " << in << "\n";
      return 3;
    }
    auto bb = CGAL::bbox_3(pts.begin(), pts.end());
    const double dx = bb.xmax() - bb.xmin();
    const double dy = bb.ymax() - bb.ymin();
    const double dz = bb.zmax() - bb.zmin();
    const double diag   = std::sqrt(dx*dx + dy*dy + dz*dz);
    const double alpha  = diag / ralpha;
    const double offset = alpha / roffset;
    Mesh wrap;
    CGAL::alpha_wrap_3(pts, alpha, offset, wrap);
    CGAL::IO::set_binary_mode(std::cout);
    CGAL::IO::write_PLY(std::cout, wrap);
    return 0;
  }

  std::string err;
  if(!wrap_one(in, ralpha, roffset, out, err)){
    std::cerr << err << "\n";
    return 3;
  }
  return 0;
}
