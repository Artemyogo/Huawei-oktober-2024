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


mt19937_64 rnd;

#define ll long long

using namespace std;

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    std:vector<Point> pts(n);
    for(auto& i : pts){
        i.x = (dbl)(rnd() % (ll)1e17)/1e15;
        i.y = (dbl)(rnd() % (ll)1e17)/1e15;
    }
    vector<Line> es;
    for(int i = 0; i < m; i++){
        int u = rnd() % n;
        int v = rnd() % n;
        while(u == v) v = rnd() % n;
        es.push_back(Line(pts[u], pts[v]));
    }
    auto [resp, reses] = build_graph(es);
    cout << resp.size() << "\n";
    for(int i = 0; i < resp.size(); i++){
        cout << i << fixed << setprecision(15)  << " " << resp[i].x << " " << resp[i].y << "\n";
    }
    cout << reses.size() << "\n";
    for(auto [a, b] : reses)
        cout << a << " " << b << "\n";
    cout << "0\n";
    
}
    
    
    
