#!/usr/bin/perl
use strict;
my ($fe, $fa);
die "Usage: $0 exact_output approximated_output\n" if(@ARGV<2);
$fe=shift @ARGV;
$fa=shift @ARGV;
open(FILEEXA,"<$fe") || die "Could not open $fe: $!\n";
open(FILEAPP,"<$fa") || die "Could not open $fa: $!\n";
my @exa=();
my @app=();
my $i=0;
while(<FILEEXA>) {
  $i++;
  next if($i<4); #skip first three lines (i.e. header)
  chomp($_);
  my ($n,$bc)=split/\t/,$_;
  push @exa,$bc;
}
my $n=$i;
$i=0;
while(<FILEAPP>) {
  $i++;
  next if($i<4); #skip first three lines (i.e. header)
  chomp($_);
  my ($n,$bc)=split/\t/,$_;
  push @app,$bc;
}
die "number of lines not equal\n" if ($i!=$n);
my ($diff, $min,$max,$ave,$aveq);
my $imin=-1;
my $imax=-1;
my $max=0;
my $min=1.e34;
for($i=0; $i<$n; $i++) {
  $diff=abs($exa[$i]-$app[$i]);
  my $rerr=$diff/$exa[$i]*100;
  warn "Relative error for node $i: $rerr\n";
  $ave+=$diff;
  $aveq+=($diff*$diff);
   if($diff>0. && $diff<$min) {
    $imin=$i;
    $min=$diff;
  }
  if($diff>$max) {
    $imax=$i;
    $max=$diff;
  }
}
$ave=$ave/$n;
my $std=sqrt(($aveq/$n)-($ave*$ave));
print "Average diff=$ave; Dev. Standard=$std; Min diff=$min (node=$imin), Max diff=$max (node=$imax)\n";
