seqname=$1
#logdir=logdir/$seqname-e120-b256-init/
#python scripts/visualize/render_root.py --testdir $logdir --first_idx 0 --last_idx 439
#logdir=logdir/$seqname-e120-b256-ft1/
#python scripts/visualize/render_root.py --testdir $logdir --first_idx 440 --last_idx 674
#logdir=logdir/$seqname-e120-b256-ft2/
#python scripts/visualize/render_root.py --testdir $logdir --first_idx 675 --last_idx 1004
logdir=logdir/$seqname-e120-b256-ft3/
python scripts/visualize/render_root.py --testdir $logdir --first_idx 0 --last_idx 439
python scripts/visualize/render_root.py --testdir $logdir --first_idx 440 --last_idx 674
python scripts/visualize/render_root.py --testdir $logdir --first_idx 675 --last_idx 1004
python scripts/visualize/render_root.py --testdir $logdir --first_idx 1005 --last_idx 1503
