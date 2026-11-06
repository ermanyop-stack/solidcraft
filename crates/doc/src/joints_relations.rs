//! Relations: joints that tie only some directions or points of two parts (B parallel to A, B's
//! point on A's plane…), so several of them between one pair of parts build up the placement the
//! way a designer thinks of it: on that face, square to that edge, 40 mm from that end. The
//! solver treats them as joints with more values; a second relation between the same parts closes
//! a loop, and the loop solver meets every relation at once (or reports the one that can't be).
//!
//! A relation is made where the parts are: its values start as the nearest that meet it, so B
//! moves no further than the relation needs.

use super::{Joint, component_bodies, component_of, least_squares, link_of, occurrence_world, resolve_origin};
use crate::{Document, IDENTITY, Mat, ModelState, mat_inverse, mat_mul};

/// Starting turns tried for each angle value: the nearest values may lie a half turn away.
const TURNS: [f64; 4] = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, -std::f64::consts::FRAC_PI_2];

/// Starts tried at most (angle values beyond the first three start at 0).
const MAX_STARTS: usize = 64;

/// The values of joint `j` that move B least from where it is now: B's joint frame as it sits
/// in A's, met as nearly as the joint allows (its origin as near as it can be to where it is,
/// and its axes, each weighed as a point a part's size along it). Used for relations and for
/// typed joints that close a loop. `None` when the origins or finite values can't be found.
pub fn relation_values(doc: &Document, st: &ModelState, j: &Joint) -> Option<Vec<f64>> {
    let world = |occ: u64| if occ == 0 { IDENTITY } else { occurrence_world(doc, occ) };
    let fa = mat_mul(&world(j.a.occurrence), &resolve_origin(doc, st, &j.a).ok()?);
    let fb = mat_mul(&world(j.b.occurrence), &resolve_origin(doc, st, &j.b).ok()?);
    // B's joint frame in A's as it is.
    let now = mat_mul(&mat_inverse(&fa)?, &fb);
    // A turn weighs as far as it moves a point B's size away.
    let size = component_of(doc, j.b.occurrence).map(|c| component_bodies(doc, st, c).iter().map(|b| b.body.size()).fold(0.0, f64::max));
    let scale = size.unwrap_or(0.0).clamp(1.0, 1e5);
    let n = j.kind.dofs().len();
    let angles: Vec<usize> = (0..n).filter(|i| j.kind.is_angle(*i)).collect();
    let mut gaps = |q: &[f64]| -> Option<Vec<f64>> {
        let m: Mat = link_of(j, q);
        let mut out = Vec::with_capacity(12);
        for (k, (row, want)) in m.iter().zip(&now).enumerate() {
            let w = if k == 3 { 1.0 } else { scale };
            out.extend(row.iter().zip(want).take(3).map(|(a, b)| (a - b) * w));
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    };
    let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let mut best: Option<(f64, Vec<f64>)> = None;
    let tried = angles.len().min(3);
    for k in 0..TURNS.len().pow(tried as u32).min(MAX_STARTS) {
        let mut q = vec![0.0; n];
        let mut code = k;
        for i in angles.iter().take(tried) {
            if let (Some(slot), Some(t)) = (q.get_mut(*i), TURNS.get(code % TURNS.len())) {
                *slot = *t;
            }
            code /= TURNS.len();
        }
        // Tolerance 0: as near as it gets; `q` keeps the best values reached.
        least_squares(&mut gaps, &mut q, &bounds, 0.0);
        let Some(c) = gaps(&q).map(|r| cost(&r)) else { continue };
        // Rotation costs grow with the square of the part's size. Keep the first of
        // numerically equal minima instead of switching symmetric poses on roundoff.
        if best.as_ref().is_none_or(|(b, _)| c < *b - 1e-12 * c.max(1.0)) {
            best = Some((c, q));
        }
    }
    // Angles within a half turn of 0.
    best.map(|(_, mut q)| {
        for i in &angles {
            if let Some(v) = q.get_mut(*i) {
                *v = (*v + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            }
        }
        q
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::joints::{JointKind, JointOrigin, Snap};
    use solidcore_record::Vec3;

    fn joint(kind: JointKind) -> Joint {
        let at = |occurrence| JointOrigin { occurrence, snap: Snap::Point { point: Vec3::ZERO, z: None, x: None } };
        Joint {
            id: 1,
            name: "R".into(),
            kind,
            a: at(0),
            b: at(1),
            values: Vec::new(),
            limits: Vec::new(),
            flip: false,
            offset: 0.0,
            angle: 0.6,
            suppressed: false,
            exprs: Default::default(),
        }
    }

    fn z_of(m: &Mat) -> Vec3 {
        Vec3::new(m[2][0], m[2][1], m[2][2])
    }

    fn fitted_angle(size: f64, now: Mat, theta: f64) -> (Mat, f64) {
        let body = solidcraft_kernel::box_solid(Vec3::ZERO, Vec3::new(size, size, size)).unwrap();
        let scale = body.size().clamp(1.0, 1e5);
        let mut st = ModelState::default();
        st.bodies.push(crate::ModelBody::new("Base".into(), body, 0));
        let mut doc = Document::default();
        doc.components.push(crate::Component { id: 1, name: "Frame".into(), ..Default::default() });
        doc.occurrences.push(crate::Occurrence {
            id: 2,
            component: 1,
            parent: 0,
            name: "Frame:1".into(),
            transform: mat_inverse(&now).unwrap(),
            grounded: false,
        });
        let at = |occurrence| JointOrigin { occurrence, snap: Snap::Frame { origin: Vec3::ZERO, z: Vec3::Z, x: Some(Vec3::X) } };
        let mut j = joint(JointKind::Angle);
        j.a = at(2);
        j.b = at(0);
        j.flip = true;
        j.angle = theta;
        let q = relation_values(&doc, &st, &j).unwrap();
        (link_of(&j, &q), scale)
    }

    fn frame_cost(got: &Mat, now: &Mat, scale: f64) -> f64 {
        got.iter()
            .zip(now)
            .enumerate()
            .map(|(column, (a, b))| {
                let weight = if column == 3 { 1.0 } else { scale };
                a.iter().zip(b).take(3).map(|(a, b)| ((a - b) * weight).powi(2)).sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn angle_fit_matches_the_analytic_nearest_pose_at_different_scales() {
        use super::super::{rot_y, rot_z, shift};
        let beta = 35.0_f64.to_radians();
        let translation = shift(3.0, -7.0, 11.0);
        let twist = rot_z(0.37);
        for size in [1.0, 30.0, 1000.0] {
            for sign in [-1.0, 1.0] {
                let now = mat_mul(&translation, &mat_mul(&rot_y(sign * beta), &twist));
                for degrees in [20.0_f64, 35.0, 35.0 + 1e-7, 35.0 + 1e-4, 60.0] {
                    let theta = degrees.to_radians();
                    let want = mat_mul(&translation, &mat_mul(&rot_y(sign * theta), &twist));
                    let (got, scale) = fitted_angle(size, now, theta);
                    // The shortest rotation changes the normal's polar angle by |theta-beta|.
                    let minimum = 8.0 * scale.powi(2) * ((theta - beta) * 0.5).sin().powi(2);
                    let cost = frame_cost(&got, &now, scale);
                    let excess = (cost - minimum).max(0.0);
                    assert!(
                        excess <= 1e-12 * minimum.max(1.0),
                        "size={size}, sign={sign}, theta={degrees}: cost={cost:.17e}, minimum={minimum:.17e}"
                    );
                    for (a, b) in got.iter().zip(&want) {
                        for (a, b) in a.iter().zip(b) {
                            assert!((a - b).abs() < 1e-7, "size={size}, sign={sign}, theta={degrees}: {got:?}, want {want:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn angle_fit_keeps_a_consistent_axis_when_the_start_is_symmetric() {
        for size in [1.0, 30.0, 1000.0] {
            let mut previous: Option<Mat> = None;
            for degrees in [140.0_f64, 150.0, 160.0, 170.0, 160.0, 150.0, 140.0] {
                let theta = degrees.to_radians();
                let (got, scale) = fitted_angle(size, IDENTITY, theta);
                let minimum = 8.0 * scale.powi(2) * (theta * 0.5).sin().powi(2);
                assert!((frame_cost(&got, &IDENTITY, scale) - minimum).abs() <= 1e-12 * minimum.max(1.0));
                if let Some(before) = previous {
                    let step = 8.0 * 5.0_f64.to_radians().sin().powi(2);
                    assert!((frame_cost(&got, &before, 1.0) - step).abs() < 1e-7, "size={size}, theta={degrees}: {before:?} -> {got:?}");
                }
                previous = Some(got);
            }
        }
    }

    /// Whatever its values, each relation keeps what it ties.
    #[test]
    fn relations_keep_what_they_tie() {
        let q = [13.0, -7.0, 21.0, 0.7, -1.9];
        let at = |kind| {
            let j = joint(kind);
            let n = kind.dofs().len();
            link_of(&j, q.get(..n).unwrap())
        };
        // Parallel: z against A's.
        assert!((z_of(&at(JointKind::Parallel)).z + 1.0).abs() < 1e-12);
        // Perpendicular: z square to A's.
        assert!(z_of(&at(JointKind::Perpendicular)).z.abs() < 1e-12);
        // Angle: 0.6 from mated.
        assert!((z_of(&at(JointKind::Angle)).z + 0.6f64.cos()).abs() < 1e-12);
        // Point on plane: the origin on A's plane.
        assert!(at(JointKind::PointOnPlane)[3][2].abs() < 1e-12);
        // Point on line: on A's z axis.
        let m = at(JointKind::PointOnLine);
        assert!(m[3][0].abs() < 1e-12 && m[3][1].abs() < 1e-12);
        // Line on plane: the origin on the plane and z along it.
        let m = at(JointKind::LineOnPlane);
        assert!(m[3][2].abs() < 1e-12 && z_of(&m).z.abs() < 1e-12);
    }
}
