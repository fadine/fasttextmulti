package fasttextmulti

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
// void load_model(char *path);
// int predict(char *query, float *prob, char *buf, int buf_sz);
import "C"
import (
	"errors"
	"unsafe"
        "strings"
)

// LoadModel - load FastText model
func LoadModel(path string) {
	C.load_model(C.CString(path))
}

// Predict - predict
func Predict(sentence string) (prob float32, labels []string, err error) {

	var cprob C.float
	var buf *C.char
	buf = (*C.char)(C.calloc(256, 1))

	ret := C.predict(C.CString(sentence), &cprob, buf, 256)

	if ret != 0 {
		err = errors.New("error in prediction")
	} else {
		label := C.GoString(buf)
                label = strings.Replace(label, "__label__", "", -1)
                labels = strings.Split(label, "=")
                labels = append(labels[:0], labels[1:]...)
		prob = float32(cprob)
	}
	C.free(unsafe.Pointer(buf))

	return prob, labels, err
}
