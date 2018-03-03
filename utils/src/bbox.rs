use std::cmp;

#[repr(C)]
pub struct Bbox {
    y_min: f32,
    x_min: f32,
    y_max: f32,
    x_max: f32,
}

impl Bbox {
    fn area(&self) -> f32 {
        if self.y_min < self.y_max && self.x_min < self.x_max {
            (self.y_max - self.y_min) * (self.x_max - self.x_min)
        } else {
            0.
        }
    }

    fn iou(&self, other: &Self) -> f32 {
        let u = Bbox {
            y_min: self.y_min.max(other.y_min),
            x_min: self.x_min.max(other.x_min),
            y_max: self.y_max.min(other.y_max),
            x_max: self.x_max.min(other.x_max),
        };
        u.area() / (self.area() + other.area() - u.area())
    }
}

pub fn non_maximum_suppression<'a, B, S>(n_bbox: usize,
                                         bbox: B,
                                         score: S,
                                         nms_thresh: f32,
                                         score_thresh: f32)
                                         -> Vec<usize>
    where B: Fn(usize) -> &'a Bbox,
          S: Fn(usize) -> f32
{
    let mut score_indices: Vec<_> = (0..n_bbox)
        .filter(|&i| score(i) >= score_thresh)
        .collect();
    score_indices.sort_unstable_by(|&i, &j| {
                                       score(j)
                                           .partial_cmp(&score(i))
                                           .unwrap_or(cmp::Ordering::Equal)
                                   });

    let mut nms_indices = Vec::new();
    for i in score_indices.into_iter() {
        if nms_indices
               .iter()
               .all(|&j| bbox(i).iou(bbox(j)) < nms_thresh) {
            nms_indices.push(i);
        }
    }

    nms_indices
}

#[cfg(test)]
mod tests {
    use super::Bbox;
    use super::non_maximum_suppression as nms;

    impl Bbox {
        fn new(y_min: f32, x_min: f32, y_max: f32, x_max: f32) -> Bbox {
            Bbox {
                y_min,
                x_min,
                y_max,
                x_max,
            }
        }
    }

    fn bbox() -> Vec<Bbox> {
        vec![Bbox::new(0., 0., 4., 4.),
             Bbox::new(1., 1., 5., 5.),
             Bbox::new(2., 1., 6., 5.),
             Bbox::new(4., 0., 8., 4.)]
    }

    #[test]
    fn test_non_maximum_supression() {
        let bbox = bbox();
        let n_bbox = bbox.len();
        let bbox = |i| &bbox[i];
        let score = |i| 1. / i as f32;

        assert_eq!(nms(n_bbox, &bbox, &score, 1., 0.), &[0, 1, 2, 3]);
        assert_eq!(nms(n_bbox, &bbox, &score, 0.5, 0.), &[0, 1, 3]);
        assert_eq!(nms(n_bbox, &bbox, &score, 0.3, 0.), &[0, 2, 3]);
        assert_eq!(nms(n_bbox, &bbox, &score, 0.2, 0.), &[0, 3]);
        assert_eq!(nms(n_bbox, &bbox, &score, 0., 0.), &[0]);
    }
}
