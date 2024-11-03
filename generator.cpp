#include <bits/stdc++.h>

using dbl = long double;
const dbl eps = 1e-9;

struct Point {
    dbl x, y;
    Point(){}
    Point(dbl x_, dbl y_): x(x_), y(y_) {}
    
    Point operator * (dbl d) const {
        return Point(x * d, y * d);
    }
    
    Point operator + (const Point & p) const {
        return Point(x + p.x, y + p.y);
    }
    
    Point operator - (const Point & p) const {
        return Point(x - p.x, y - p.y);
    }
    
    dbl cross (const Point & p) const {
        return x * p.y - y * p.x;
    }
    
    dbl cross (const Point & p, const Point & q) const {
        return (p - *this).cross(q - *this);
    }
    
    dbl dot (const Point & p) const {
        return x * p.x + y * p.y;
    }
    
    dbl dot (const Point & p, const Point & q) const {
        return (p - *this).dot(q - *this);
    }
    
    bool operator < (const Point & p) const {
        if (fabs(x - p.x) < eps) {
            if (fabs(y - p.y) < eps) {
                return false;
            } else {
                return y < p.y;
            }
        } else {
            return x < p.x;
        }
    }
    
    bool operator == (const Point & p) const {
        return fabs(x - p.x) < eps && fabs(y - p.y) < eps;
    }
    
    bool operator >= (const Point & p) const {
        return !(*this < p);
    }
};

bool eq(dbl x, dbl y) {return abs(x - y) < eps; }
int sgn(const dbl& x) { return eq(x, 0) ? 0 : x <= 0 ? -1 : 1; }

struct Line{
    Point p[2];
    
    Line(Point l, Point r){p[0] = l; p[1] = r;}
    Point& operator [](const int & i){return p[i];}
    const Point& operator[](const int & i)const{return p[i];}
    Line(const Line & l){
        p[0] = l.p[0]; p[1] = l.p[1];
    }
    Point getOrth()const{
        return Point(p[1].y - p[0].y, p[0].x - p[1].x);
    }
    bool hasPointLine(const Point & t)const{
        return std::fabs(p[0].cross(p[1], t)) < eps;
    }
    bool hasPointSeg(const Point & t)const{
        return hasPointLine(t) && t.dot(p[0], p[1]) < eps;
    }
};

std::vector<Point> interLineLine(Line l1, Line l2){
    if(std::fabs(l1.getOrth().cross(l2.getOrth())) < eps){
        if(l1.hasPointLine(l2[0]))return {l1[0], l1[1]};
        else return {};
    }
    Point u = l2[1] - l2[0];
    Point v = l1[1] - l1[0];
    dbl s = u.cross(l2[0] - l1[0])/u.cross(v);
    return {Point(l1[0] + v * s)};
}

std::vector<Point> interSegSeg(Line l1, Line l2){
    if (l1[0] == l1[1]) {
        if (l2[0] == l2[1]) {
            if (l1[0] == l2[0])
                return {l1[0]};
            else
                return {};
        } else {
            if (l2.hasPointSeg(l1[0]))
                return {l1[0]};
            else
                return {};
        }
    }
    if (l2[0] == l2[1]) {
        if (l1.hasPointSeg(l2[0]))
            return {l2[0]};
        else
            return {};
    }
    auto li = interLineLine(l1, l2);
    if (li.empty())
        return li;
    if (li.size() == 2) {
        if (l1[0] >= l1[1])
            std::swap(l1[0], l1[1]);
        if (l2[0] >= l2[1])
            std::swap(l2[0], l2[1]);
        std::vector<Point> res(2);
        if (l1[0] < l2[0])
            res[0] = l2[0];
        else
            res[0] = l1[0];
        if (l1[1] < l2[1])
            res[1] = l1[1];
        else
            res[1] = l2[1];
        if (res[0] == res[1])
            res.pop_back();
        if (res.size() == 2u && res[1] < res[0])
            return {};
        else
            return res;
    }
    Point cand = li[0];
    if (l1.hasPointSeg(cand) && l2.hasPointSeg(cand))
        return {cand};
    else
        return {};
}

std::pair<std::vector<Point>, std::vector<std::pair<int, int> > > build_graph(std::vector<Line> segments) {
    std::vector<Point> p;
    std::vector<pair<int, int> > adj;
    std::map<std::pair<int64_t, int64_t>, size_t> point_id;
    auto get_point_id = [&](Point pt) {
        auto repr = std::make_pair(
                                   int64_t(std::round(pt.x * 1000000000) + 1e-6),
                                   int64_t(std::round(pt.y * 1000000000) + 1e-6)
                                   );
        if (!point_id.count(repr)) {
            size_t id = point_id.size();
            point_id[repr] = id;
            p.push_back(pt);
            return id;
        } else {
            return point_id[repr];
        }
    };
    for (size_t i = 0; i < segments.size(); i++) {
        std::vector<size_t> curr = {
            get_point_id(segments[i][0]),
            get_point_id(segments[i][1])
        };
        for (size_t j = 0; j < segments.size(); j++) {
            if (i == j)
                continue;
            auto inter = interSegSeg(segments[i], segments[j]);
            for (auto pt: inter) {
                curr.push_back(get_point_id(pt));
            }
        }
        std::sort(curr.begin(), curr.end(), [&](size_t l, size_t r) { return p[l] < p[r]; });
        curr.erase(std::unique(curr.begin(), curr.end()), curr.end());
        for (size_t j = 0; j + 1 < curr.size(); j++) {
            adj.push_back({curr[j], curr[j + 1]});
        }
    }
    return {p, adj};
}


//mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());

#define ll long long

using namespace std;

struct edge {
    Point l, r;
    int up;
    edge() {}
    edge(Point a, Point b, int sw){
        l = a;
        r = b;
        up = sw;
    }
};

inline bool edge_cmp(const edge& e1, const edge&e2){
    Point a = e1.l, b = e1.r;
    Point c = e2.l, d = e2.r;
    int val = sgn(a.cross(b, c)) + sgn(a.cross(b, d));
    if (val != 0)
        return val > 0;
    val = sgn(c.cross(d, a)) + sgn(c.cross(d, b));
    return val < 0;
}



struct event{
    int type; //-1 — del, 0 — get, 1 - add
    dbl x;
    edge e;
};

inline bool operator <(const event& e1, const event& e2){
    if(!eq(e1.x, e2.x))
        return e1.x < e2.x;
    if(abs(e1.type) != abs(e2.type))
        return abs(e1.type) < abs(e2.type);
    else if(e1.type != e2.type)
        return e1.type < e2.type;
    else return edge_cmp(e1.e, e2.e);
}


void dfs(int v, int p, vector<vector<int> >& g, vector<bool>& used, vector<int>& tin, vector<int>& tout, vector<int>& fup, int& timer, map<pair<int, int>, bool>& del) {
    used[v] = true;
    tin[v] = fup[v] = timer++;
    for (size_t i=0; i<g[v].size(); ++i) {
        int to = g[v][i];
        if (to == p)  continue;
        if (used[to])
            fup[v] = min (fup[v], tin[to]);
        else {
            dfs(to, v, g, used, tin, tout, fup, timer, del);
            fup[v] = min (fup[v], fup[to]);
            if (fup[to] > tin[v])
                del[{v, to}] = del[{to, v}] = 1;
        }
    }
}

bool point_in_polygon(Point point, vector<Point> polygon)
{
    int num_vertices = polygon.size();
    double x = point.x, y = point.y;
    bool inside = false;
    
    // Store the first point in the polygon and initialize
    // the second point
    Point p1 = polygon[0], p2;
    
    // Loop through each edge in the polygon
    for (int i = 1; i <= num_vertices; i++) {
        // Get the next point in the polygon
        p2 = polygon[i % num_vertices];
        
        // Check if the point is above the minimum y
        // coordinate of the edge
        if (y > min(p1.y, p2.y)) {
            // Check if the point is below the maximum y
            // coordinate of the edge
            if (y <= max(p1.y, p2.y)) {
                // Check if the point is to the left of the
                // maximum x coordinate of the edge
                if (x <= max(p1.x, p2.x)) {
                    // Calculate the x-intersection of the
                    // line connecting the point to the edge
                    double x_intersection
                    = (y - p1.y) * (p2.x - p1.x)
                    / (p2.y - p1.y)
                    + p1.x;
                    
                    // Check if the point is on the same
                    // line as the edge or to the left of
                    // the x-intersection
                    if (p1.x == p2.x
                        || x <= x_intersection) {
                        // Flip the inside flag
                        inside = !inside;
                    }
                }
            }
        }
        
        // Store the current point as the first point for
        // the next iteration
        p1 = p2;
    }
    
    // Return the value of the inside flag
    return inside;
}

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int t = atoi(argv[3]);
    mt19937_64 rnd(atoi(argv[4]));
    std:vector<Point> pts(n);
    map<pair<dbl, dbl>, bool> cvis;
    for(auto& i : pts){
        do{
            i.x = (dbl)(rnd() % (ll)10000)/1000;
            i.y = (dbl)(rnd() % (ll)10000)/1000;
        } while(cvis[{i.x, i.y}]);
        cvis[{i.x, i.y}] = 1;
    }
    vector<Line> es;
    map<pair<int, int>, bool> vis;
    for(int i = 0; i < m; i++){
        int u = rnd() % n;
        int v = rnd() % n;
        while(u == v) v = rnd() % n;
        if(vis[{u, v}]) continue;
        vis[{u, v}] = vis[{v, u}] = 1;
        es.push_back(Line(pts[u], pts[v]));
    }
    ofstream fout((string("../graphs/graph_data") + argv[4] + ".txt").c_str());
    auto [resp, reses] = build_graph(es);
    cout << resp.size() << "\n";
    fout << resp.size() << "\n";

    for(int i = 0; i < resp.size(); i++){
        cout << i << fixed << setprecision(15)  << " " << resp[i].x << " " << resp[i].y << "\n";
        fout << i << fixed << setprecision(15)  << " " << resp[i].x << " " << resp[i].y << "\n";

    }
    cout << reses.size() << "\n";
    fout << reses.size() << "\n";
    for(auto [a, b] : reses){
        cout << a << " " << b << "\n";
        fout << a << " " << b << "\n";
    }
    vector<Point> users;
    for(auto& i : users){
        
        i.x = (dbl)(rnd() % (ll)100)/10;
        i.y = (dbl)(rnd() % (ll)100)/10;
    }
    point_location(resp, reses, users);
    cout << users.size() << "\n";
    fout << users.size() << "\n";

    for(int i = 0; i < users.size(); i++){
        cout << i << " " << users[i].x << " " << users[i].y << "\n";
        fout << i << " " << users[i].x << " " << users[i].y << "\n";
    }
    fout.close();
    
}
    
    
    
