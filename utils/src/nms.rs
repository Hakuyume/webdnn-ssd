use num_traits::Num;

use Rect;

pub fn non_maximum_suppression_by<'a, T, R, I, F>(iter: I,
                                                  f: F,
                                                  thresh: T)
                                                  -> impl 'a + Iterator<Item = I::Item>
    where T: 'a + Copy + PartialOrd + Num,
          R: 'a + Rect<T>,
          I: 'a + Iterator,
          F: 'a + Fn(&I::Item) -> R
{
    iter.scan(Vec::new(), move |selected, item| {
            let rect = f(&item);
            if selected.iter().all(|r| rect.iou(r) < thresh) {
                selected.push(rect);
                Some(Some(item))
            } else {
                Some(None)
            }
        })
        .filter_map(|item| item)
}

pub fn non_maximum_suppression<'a, T, R, I>(iter: I, thresh: T) -> impl 'a + Iterator<Item = R>
    where T: 'a + Copy + PartialOrd + Num,
          R: 'a + Copy + Rect<T>,
          I: 'a + Iterator<Item = R>
{
    non_maximum_suppression_by(iter, |&r| r, thresh)
}

#[cfg(test)]
mod tests {
    use super::Rect;
    use super::non_maximum_suppression;
    use super::non_maximum_suppression_by;

    #[derive(Debug, PartialEq)]
    struct R(f32, f32, f32, f32);
    impl<'a> Rect<f32> for &'a R {
        fn x_min(&self) -> f32 {
            self.0
        }
        fn y_min(&self) -> f32 {
            self.1
        }
        fn x_max(&self) -> f32 {
            self.2
        }
        fn y_max(&self) -> f32 {
            self.3
        }
    }

    fn check<F>(f: F)
        where F: Fn(&[R], f32, &[usize])
    {
        let rects = [R(0., 0., 4., 4.),
                     R(1., 1., 5., 5.),
                     R(2., 1., 6., 5.),
                     R(4., 0., 8., 4.)];

        f(&rects, 1., &[0, 1, 2, 3]);
        f(&rects, 0.5, &[0, 1, 3]);
        f(&rects, 0.3, &[0, 2, 3]);
        f(&rects, 0.2, &[0, 3]);
        f(&rects, 0., &[0]);
    }

    #[test]
    fn test_non_maximum_supression() {
        check(|rects, thresh, indices| {
                  let result: Vec<_> = non_maximum_suppression(rects.iter(), thresh).collect();
                  let expect: Vec<_> = indices.iter().map(|&i| &rects[i]).collect();
                  assert_eq!(result, expect);
              });
    }

    #[test]
    fn test_non_maximum_supression_by() {
        check(|rects, thresh, indices| {
                  let result: Vec<_> =
                      non_maximum_suppression_by(0..rects.len(), |&i| &rects[i], thresh).collect();
                  assert_eq!(result, indices);
              });
    }
}
