create or replace function public.dot_product(a cube, b cube)
returns double precision
language plpgsql
as
$$
declare
   dim integer;
   sum_a double precision := 0.0;
   sum_b double precision := 0.0;
begin
   select cube_dim(a) into dim;
    DO
    $do$
    BEGIN
       FOR i IN 1..dim LOOP
          curr_a := sum_a;
          curr_b := sum_b;
          sum_a := curr_a + cube_ll_coord(a, i);
          sum_b := curr_b + cube_ll_coord(b, i);
       END LOOP;
    END
    $do$;

   return sum_a * sum_b;
end;
$$;